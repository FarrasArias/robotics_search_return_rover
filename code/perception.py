import numpy as np
import cv2

# Identify pixels above the threshold
# Threshold of RGB > 160 does a nice job of identifying ground pixels only
def color_thresh(img, rgb_thresh=(160, 160, 160)):
    # Create an array of zeros same xy size as img, but single channel
    color_select = np.zeros_like(img[:,:,0])
    # Require that each pixel be above all three threshold values in RGB
    # above_thresh will now contain a boolean array with "True"
    # where threshold was met
    above_thresh = (img[:,:,0] > rgb_thresh[0]) \
                & (img[:,:,1] > rgb_thresh[1]) \
                & (img[:,:,2] > rgb_thresh[2])
    # Index the array of zeros with the boolean array and set to 1
    color_select[above_thresh] = 1
    # Return the binary image
    return color_select

def find_rocks(img, rgb_thresh=(110,110,50)):
    color_select = np.zeros_like(img[:,:,0])
    above_thresh = (img[:,:,1]>rgb_thresh[0]) & (img[:,:,1]>rgb_thresh[1]) & (img[:,:,2]<rgb_thresh[2])
    color_select[above_thresh] = 1
    return color_select

# Define a function to convert from image coords to rover coords
def rover_coords(binary_img):
    # Identify nonzero pixels
    ypos, xpos = binary_img.nonzero()
    # Calculate pixel positions with reference to the rover position being at the 
    # center bottom of the image.  
    x_pixel = -(ypos - binary_img.shape[0]).astype(np.float)
    y_pixel = -(xpos - binary_img.shape[1]/2 ).astype(np.float)
    return x_pixel, y_pixel


# Define a function to convert to radial coords in rover space
def to_polar_coords(x_pixel, y_pixel):
    # Convert (x_pixel, y_pixel) to (distance, angle) 
    # in polar coordinates in rover space
    # Calculate distance to each pixel
    dist = np.sqrt(x_pixel**2 + y_pixel**2)
    # Calculate angle away from vertical for each pixel
    angles = np.arctan2(y_pixel, x_pixel)
    return dist, angles

# Define a function to map rover space pixels to world space
def rotate_pix(xpix, ypix, yaw):
    # Convert yaw to radians
    yaw_rad = yaw * np.pi / 180
    xpix_rotated = (xpix * np.cos(yaw_rad)) - (ypix * np.sin(yaw_rad))
                            
    ypix_rotated = (xpix * np.sin(yaw_rad)) + (ypix * np.cos(yaw_rad))
    # Return the result  
    return xpix_rotated, ypix_rotated

def translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale): 
    # Apply a scaling and a translation
    xpix_translated = (xpix_rot / scale) + xpos
    ypix_translated = (ypix_rot / scale) + ypos
    # Return the result  
    return xpix_translated, ypix_translated


# Define a function to apply rotation and translation (and clipping)
# Once you define the two functions above this function should work
def pix_to_world(xpix, ypix, xpos, ypos, yaw, world_size, scale):
    # Apply rotation
    xpix_rot, ypix_rot = rotate_pix(xpix, ypix, yaw)
    # Apply translation
    xpix_tran, ypix_tran = translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale)
    # Perform rotation, translation and clipping all at once
    x_pix_world = np.clip(np.int_(xpix_tran), 0, world_size - 1)
    y_pix_world = np.clip(np.int_(ypix_tran), 0, world_size - 1)
    # Return the result
    return x_pix_world, y_pix_world

#functions to compare pixels that have been mapped to the current Rover vision
def inv_rotate_pix(xrot, yrot, yaw):
     yaw_rad = yaw * np.pi / 180
     xpix = xrot * np.cos(yaw_rad) + yrot * np.sin(yaw_rad)
     ypix = -xrot * np.sin(yaw_rad) + yrot * np.cos(yaw_rad)
     return xpix,ypix

def inv_translate_pix(world_x,world_y,xpos,ypos,scale):
     xpix_rot = (world_x - xpos)*scale
     ypix_rot = (world_y - ypos)*scale
     return xpix_rot,ypix_rot

def world_to_pix(world_x, world_y, xpos, ypos, scale, yaw):
    xpix_rot, ypix_rot = inv_translate_pix(world_x, world_y, xpos, ypos, scale)
    xpix, ypix = inv_rotate_pix(xpix_rot, ypix_rot, yaw)
    return xpix, ypix
    

# Define a function to perform a perspective transform
def perspect_transform(img, src, dst):
           
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))# keep same size as input image
    mask = cv2.warpPerspective(np.ones_like(img[:,:,0]), M, (img.shape[1], img.shape[0]))
    return warped, mask


# Apply the above functions in succession and update the Rover state accordingly
def perception_step(Rover):
    # Perform perception steps to update Rover()
    # TODO: 
    # NOTE: camera image is coming to you in Rover.img
    # 1) Define source and destination points for perspective transform
    
    dst_size = 5
    bottom_offset = 5
    
    source = np.float32([[14, 140], [301 ,140],[200, 96], [118, 96]])
    destination = np.float32([[Rover.img.shape[1]/2 - dst_size, Rover.img.shape[0] - bottom_offset],
                  [Rover.img.shape[1]/2 + dst_size, Rover.img.shape[0] - bottom_offset],
                  [Rover.img.shape[1]/2 + dst_size, Rover.img.shape[0] - 2*dst_size - bottom_offset], 
                  [Rover.img.shape[1]/2 - dst_size, Rover.img.shape[0] - 2*dst_size - bottom_offset],
                  ])
    # 2) Apply perspective transform
    warped, mask = perspect_transform(Rover.img, source, destination)
    
    # 3) Apply color threshold to identify navigable terrain/obstacles/rock samples
    threshed = color_thresh(warped)
    obs_map = np.absolute((np.float32(threshed)-1)*mask)
    rock_map = find_rocks(warped)
    
    # 4) Update Rover.vision_image (this will be displayed on left side of screen)
        # Example: Rover.vision_image[:,:,0] = obstacle color-thresholded binary image
        #          Rover.vision_image[:,:,1] = rock_sample color-thresholded binary image
        #          Rover.vision_image[:,:,2] = navigable terrain color-thresholded binary image
    Rover.vision_image[:,:,0] = obs_map * 255
    Rover.vision_image[:,:,1] = rock_map * 255
    Rover.vision_image[:,:,2] = threshed * 255
    
    # 5) Convert map image pixel values to rover-centric coords
    xpix, ypix = rover_coords(threshed)
    obsxpix, obsypix = rover_coords(obs_map)
    
    #grab 3 parts of the obstacle map to check if we have a rock
    left_obs_map = obs_map[70:,120:140]
    center_obs_map = obs_map[70:,140:180]
    right_obs_map = obs_map[70:,180:200]
    
    #sum total nunber of obstacle pixels to compare
    center_count = (center_obs_map == 1).sum()
    right_count = (right_obs_map == 1).sum()
    left_count = (left_obs_map == 1).sum()
    
    # 6) Convert rover-centric pixel values to world coordinates
    
    #get info about map and rover
    xpos = Rover.pos[0]
    ypos = Rover.pos[1]
    yaw = Rover.yaw
    world_size = Rover.worldmap.shape[0]
    scale = 2 * dst_size
    
    x_world, y_world = pix_to_world(xpix, ypix, xpos, ypos, yaw, world_size, scale)
    obs_x_world, obs_y_world = pix_to_world(obsxpix, obsypix, xpos, ypos, yaw, world_size, scale)
    
    if rock_map.any():
        rock_x, rock_y = rover_coords(rock_map)
        rock_x_world, rock_y_world = pix_to_world(rock_x, rock_y, xpos, ypos, yaw, world_size, scale)
        if Rover.pitch%359.5 <= 0.05 and Rover.roll%358.5 < 1.5:
            Rover.worldmap[rock_x_world, rock_y_world, 2] += 1
        #print ("here",Rover.nav_angles)

    # 7) Update Rover worldmap (to be displayed on right side of screen)
        # Example: Rover.worldmap[obstacle_y_world, obstacle_x_world, 0] += 1
        #          Rover.worldmap[rock_y_world, rock_x_world, 1] += 1
        #          Rover.worldmap[navigable_y_world, navigable_x_world, 2] += 1
        
    if Rover.pitch%359.5 <= 0.05 and Rover.roll%358.5 < 1.5:
        Rover.worldmap[obs_y_world, obs_x_world, 0] += 1
        Rover.worldmap[y_world, x_world, 2] += 10


    # 8) Convert rover-centric pixel positions to polar coordinates
    # Update Rover pixel distances and angles
        # Rover.nav_dists = rover_centric_pixel_distances
        # Rover.nav_angles = rover_centric_angles
        
    #if there's any pixel in the rock_map, change the Rover valid nav angles to these rock angles
    if rock_map.any():
        Rover.stop_forward = 0
        Rover.go_forward = 0
        dist, angles = to_polar_coords(rock_x, rock_y)
        Rover.nav_dists = dist
        Rover.nav_angles = angles
        #print ("sample",Rover.near_sample)
    #if there are more obstacle pixels in the center than in both sides, it's probably a rock
    #grab left part of the map and use it to define nav angles
    elif center_count > left_count*3 and center_count > right_count*3:
        print("Rock Ahead!")
        new_img = np.zeros_like(Rover.img[:,:,0])
        new_img[:,:150] = obs_map[:,:150]
        cut_xpix, cut_ypix = rover_coords(new_img)
        cut_xpix *= 4
        cut_ypix *= 4
        cut_dist, cut_angles = to_polar_coords(cut_xpix, cut_ypix)
        Rover.nav_dists = cut_dist
        Rover.nav_angles = cut_angles

    #if there isn't any pick up rocks and no obs rocks ahead, revert to previous
    #variable values and navigate
    else:
        Rover.stop_forward = 150
        Rover.go_forward = 1000
        
        dist_nan, angles_nan = to_polar_coords(xpix, ypix)

        #make map to check all parts in the obs world with pixel values over 250
        thresh_world = np.zeros_like(Rover.worldmap[:,:,0])
        thresh_px = Rover.worldmap[:,:,0] > 250
        thresh_world[thresh_px] = 1
        #Obtain only the part we're dealing with right now
        specific_thresh_px = thresh_world[obs_y_world, obs_x_world]
        #Get x and y values for the map
        thresh_x = specific_thresh_px[0]
        thresh_y = specific_thresh_px[1]
        #make reverse steps to get the pixels we have mapped before in rover coords
        xpix_thresh, ypix_thresh = world_to_pix(thresh_x, thresh_y, ypos, ypos, scale, yaw)
        #separate them with the ones we've mapped to the ones that are under the threshold
        xpix_ones = xpix_thresh > 0
        ypix_ones = ypix_thresh > 0
        xpix_zeros = xpix_thresh < 1
        ypix_zeros = ypix_thresh < 1
        #divide mapped pixels by half and multiply unseen pixels by 2, to bias the
        #mean angle towards were there are more unmapped pixels
        xpix[xpix_ones] *= 0.5
        ypix[ypix_ones] *= 0.5
        xpix[xpix_zeros] *= 2
        ypix[ypix_zeros] *= 2
        #transform them to polar coords
        dist, angles = to_polar_coords(xpix, ypix)
        Rover.nav_dists = dist
        Rover.nav_angles = angles

    return Rover