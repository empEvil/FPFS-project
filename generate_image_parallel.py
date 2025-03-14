import batsim
import numpy as np

def generate_images(i, nn, scale, gal, records, psf):

    i = i // 4

    # rescale flux to match LSST
    mag_zero = 30
    flux = 10 ** ((mag_zero - records['mag_auto'][i]) / 2.5)
    gal = gal.withFlux(flux)
    
    # Get HLR for galaxy to create IA transform
    if records['use_bulgefit'][i]:
        hlr = records['hlr'][i][2]
    else:
        hlr = records['hlr'][i][0]

    # Create IA transform
    IATransform = batsim.IaTransform(
        scale=scale,
        hlr=hlr,
        A=0.00136207,
        beta=0.82404653, # best first Georgiou19+
        phi = np.radians(0),
        clip_radius=None # clip the transform at 5*hlr to prevent edge effects
    )

    # Convolve with PSF and pixel response
    gal_img = batsim.simulate_galaxy(
        ngrid=nn,
        pix_scale=scale,
        gal_obj=gal,
        transform_obj=IATransform,
        psf_obj=psf,
        draw_method="auto"
    )

    return gal_img