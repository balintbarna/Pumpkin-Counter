.. include:: <isonum.txt>
Exercise 6 - GSD
________________

If we try to measure a size on an image, the unit will be in pixels.
That is not very useful if we want to get information of pumpkin sizes or number of pumpkins per square meter.
To connect image units to physical units, we calculate a GSD ratio in mm/pixel.

The following source code was used for this task:

::

    pixels = img.shape
    alpha = math.radians(fov_deg / 2)
    width = height_meters * 2 * math.tan(alpha)
    ratio = width / pixels[0]
    height = ratio * pixels[1]
    ratio *= 1000 # convert to mm/pixel

These were our parameters:

- Relative height: 54.2 m
- Field-of-View: 73.7 |deg|

We got the following results:

- Image width: 81.24119485997007 m
- Image height: 45.69817210873316 m
- Ratio: 14.846709586982833 mm/pixel
