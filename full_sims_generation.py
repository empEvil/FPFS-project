import galsim
import batsim
import numpy as np
import fitsio 
import os
import argparse

from multiprocessing import Pool, cpu_count 
from time import time


def _init_worker(scale, psf, pixel_side, rng, flux_var, zero_mag):
    global _STATE
    _STATE["scale"] = scale
    _STATE['psf'] = psf
    _STATE['pixel_side'] = pixel_side
    _STATE['rng'] = rng
    _STATE['flux_var'] = flux_var
    _STATE['zero_mag'] = zero_mag

def generate_image(galaxy, mag, hlr):
    s = _STATE
    # prepping the galaxy
    flux = 10**((mag-s['zero_mag'])/2.5)
    flux *= s['rng'].uniform(1-s['flux_var'], 1+s['flux_var'])
    gal = galaxy.withFlux(flux)
    rot_ang = np.pi *s['rng'].random()
    gal.rotate(rot_ang*galsim.radians)

    # Creating the IA transform
    IATransform = batsim.IaTransform(
        scale = s['scale'],
        hlr = hlr,
        A = 0.00136207,
        beta = 0.82404653,
        phi = np.radians(0),
        clip_radius=5
    )

    #prepping stamp
    stamp = galsim.ImageF(s['pixel_side'], s['pixel_side'], scale=s['scale'])
    xmin = 1
    xmax = s['pixel_side'] + 1
    ymin = 1
    ymax = s['pixel_side'] + 1

    # Doing the simulation
    gal_img = batsim.simulate_galaxy(
        ngrid = s['pixel_side'],
        pix_scale = s['scale'],
        gal_obj = gal, 
        transform_obj = IATransform,
        psf_obj=s['psf'],
        draw_method='auto'
    )

    bounds = galsim.BoundsI(xmin, xmax, ymin, ymax)
    sub_image = galsim.Image(gal_img, scale=s['scale'])
    stamp[bounds] = sub_image

    return stamp


def main(args):
    """
    Generating random LSST-Like images from COSMOS parameters,
    generating galaxies with IA and PSF
    """

    t0 = time()

    n_gals = args.n_gals  # number of galaxies
    pixel_side = args.pixel_side  # pixels per side
    scale = args.scale  #pixel scale in arcseconds
    save_dir = args.save_dir  # save location
    batch_process = args.batch_process  # multiprocessing
    rng = np.random.default_rng(seed=args.seed)  # the seed for the random algorithm
    seeing = args.seeing  # PSF fwhm
    flux_var = args.flux_var  # limit of flux change in % 
    zero_mag = args.zero_mag  # zero magnitude

    cosmos_cat = galsim.COSMOSCatalog()

    gal_indx = rng.choice(len(cosmos_cat), n_gals)  # Should we let the same galaxy be pulled multiple times?
    records = cosmos_cat.getParametricRecord(gal_inds)
    galaxies = cosmos_cat.makeGalaxy(index=gal_indx, gal_type='parametric')
    ids = records['IDENT']
    mags = records['mag_auto']
    hlr = [a if c else b for a,b,c in zip(records['hlr'][:,2], records['hlr'][:,0], records['use_bulgefit'])]
    
    print(f"Generating {n_gals} images, {pixel_side}x{pixel_side} pixels, scale {scale} arcsex/pixel")
    psf = galsim.Moffat(beta=2.5, fwhm=seeing, trunc=seeing*4)

    os.makedirs(save_dir, exist_ok=True)

    # Creating fits file to save images
    fits = fitsio.FITS(
        os.path.join(save_dir, f'COSMOS_ngals={n_gals}_noisy.fits'),
        'rw',
        clobber=True
        )
    
    if batch_process == True:
        print("Batch processing images")
        with Pool(initializer=_init_worker, 
                  initargs=(scale, psf, pixel_side, rng, flux_var, zero_mag)) as pool:
            stamps = pool.starmap(generate_image, zip(galaxies, mags, hlr))
    else:
        raise ValueError("you should multriprocess")
    
    fits.write(None, header={}, extname='primary')

    for i, stamp in enumerate(stamps):
        fits.write(stamp.array, header={'IDENT': ids[i]}, extname=f'GALAXY_{i}')
    
    psf_image = psf.drawImage(nx=pixel_side, ny=pixel_side, scale=scale).array
    fits.write(psf_image, header={}, extname='PSF')

    fits.close()

    print(f"Simulated {n_gals} galaxies in {time() -t0:.2f} seconds")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate noiseless LSST-like images from COSMOS galaxies')
    parser.add_argument('--n_gals', type='int', default=1, help='Number of galaxies to simulate')
    parser.add_argument('--pixel_side', type='int', default=64, help='Number of pixels on a side')
    parser.add_argument('--scale', type=float, default=0.2, help='Pixel scale in arcseconds')
    parser.add_argument('--save_dir', type=str, default='.', help='Directory to save images')
    parser.add_argument('--batch_process', type=str, default='True', help='Batch process the images')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    parser.add_argument('--seeing', type=float, default=0.8, help='PSF FWHM')
    parser.add_argument('--flux_var', type=float, default=0.25, help='how much we change the flux by')
    parser.add_argument('--zero_mag', type=float, default=30, help='Zeropoint magnitude')

    args = parser.parse_args()

    main(args)