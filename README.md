# PixelReCreator
Take a shitty jpg image with some Pixel Art ilustration and restore it in a 1:1 pixel ratio png with the user palette color

## Libraries needed

PIL, numpy, cv2

## Parameters

**cuts** = 1              # just in case you want to check an source image by parts

**pixel_density** = 8     # aprox pixel density of the source image

**user_scale** = 1        # show an escaled version just for the user accessibility

**check_metod** = None    # average | mean | None

**name_input** = 'c:/Users/User/Desktop/PixelReCreator/example_in.jpg'

**name_output** = 'c:/Users/User/Desktop/PixelReCreator/example_out_palette.png'

**name_palette** = 'c:/Users/User/Desktop/PixelReCreator/brick_palette.png'      # 1x1 color palette png | None
