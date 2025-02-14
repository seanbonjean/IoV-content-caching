import utils

CENTER_LOCATION = (116.36032115, 39.911045075)

# utils.batch_GPStoXY("mbs_gps.txt", "mbs_xy.txt",
#                     CENTER_LOCATION[0], CENTER_LOCATION[1], lon_pos=0, lat_pos=1, src_sep='\t')
# utils.batch_GPStoXY("rsu_gps.txt", "rsu_xy.txt",
#                     CENTER_LOCATION[0], CENTER_LOCATION[1], lon_pos=0, lat_pos=1, src_sep='\t')
#
# utils.batch_GPStoXY("rsu_added_gps.txt", "rsu_added_xy.txt",
#                     CENTER_LOCATION[0], CENTER_LOCATION[1], lon_pos=0, lat_pos=1, src_sep='\t')

utils.batch_GPStoXY("rsu_added_gps_v9.txt", "rsu_added_xy_v9.txt",
                    CENTER_LOCATION[0], CENTER_LOCATION[1], lon_pos=0, lat_pos=1, src_sep=' ')
