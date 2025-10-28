import galsim
import batsim
import numpy as np
import fitsio 
import os
import argparse
import math
import gc

from multiprocessing import Pool, cpu_count 
from time import time

_STATE = {}

def _init_worker(scale, psf, pixel_side, n_pixel_side, rng, var, zero_mag,
                 rand_flux, rand_size):
    global _STATE
    _STATE["scale"] = scale
    _STATE['psf'] = psf
    _STATE['pixel_side'] = pixel_side
    _STATE['n_pixel_side'] = n_pixel_side
    _STATE['rng'] = rng
    _STATE['var'] = var
    _STATE['zero_mag'] = zero_mag
    _STATE['rand_flux'] = rand_flux
    _STATE['rand_size'] = rand_size

def generate_image(galaxy, mag, hlr):
    """ The actual function that generates the galaxy image via batsim
    arguments:
    ----------
    galaxy: `galaxy.object`
        Galsim galaxy object (parametric)
    mag: `float`
        the magnitude of the galaxy from the records
    hlr: `float`
    """
    s = _STATE
    # prepping the galaxy
    flux = 10**((mag-s['zero_mag'])/2.5)
    if s['rand_flux'] == True:
        flux *= s['rng'].uniform(1-s['var'], 1+s['var'])
    gal = galaxy.withFlux(flux)
    rot_ang = np.pi *s['rng'].random()
    gal.rotate(rot_ang*galsim.radians)
    if s['rand_size'] == True:
        gal.expand(s['rng'].uniform(1-s['var'], 1+s['var']))

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
    stamp = galsim.ImageF(s['n_pixel_side'], s['n_pixel_side'], scale=s['scale'])

    # Doing the simulation
    gal_img = batsim.simulate_galaxy(
        ngrid = s['n_pixel_side'],  # n_pixel or pixel?
        pix_scale = s['scale'],
        gal_obj = gal,
        transform_obj = IATransform,
        psf_obj=s['psf'],
        draw_method='auto'
    )
    sub_image = galsim.Image(gal_img, scale=s['scale'])
    bound_size = max(sub_image.bounds.xmax, sub_image.bounds.ymax)
    padding = s['n_pixel_side'] - bound_size
    if padding < 0:
        raise ValueError('the stamp sizes are too small')
    offset = padding // 2
    bounds = galsim.BoundsI(offset+1, offset+bound_size,
                            offset+1, offset+bound_size)
    sub_image = galsim.Image(gal_img, scale=s['scale'])
    #print(stamp, bounds, sub_image)
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
    var = args.var  # limit of flux change in % 
    zero_mag = args.zero_mag  # zero magnitude
    randomizer = args.random_sizeflux
    if randomizer=='both':
        rand_size = True
        rand_flux = True
    elif randomizer=='size':
        rand_size = True
        rand_flux = False
    elif randomizer=='flux':
        rand_size = False
        rand_flux = True
    elif randomizer=='None':
        rand_size = False
        rand_flux = False
    else:
        raise ValueError("You should select if you want to randomize flux, size or both")
    
    if batch_process == 'True':
        batch_process = True

    cosmos_cat = galsim.COSMOSCatalog()

    gal_indx = rng.choice(len(cosmos_cat), n_gals)  # Should we let the same galaxy be pulled multiple times?
    records = cosmos_cat.getParametricRecord(gal_indx)
    galaxies = cosmos_cat.makeGalaxy(index=gal_indx, gal_type='parametric')
    ids = records['IDENT']
    mags = records['mag_auto']
    hlr = [a if c else b for a,b,c in zip(records['hlr'][:,2], records['hlr'][:,0], records['use_bulgefit'])]
    

    # WE NEED TO EXPAND OUR IMAGES
    # we start with pixel_side x pixel_side galaxies, and since we vary their size, we need to scale up our images
    n_pixel_side = math.ceil(pixel_side * (1.0+2*var)) +2  # so we pad our image size by 2 times the variance in size
    print(f"Generating {n_gals} images, {n_pixel_side}x{n_pixel_side} pixels, scale {scale} arcsex/pixel")
    psf = galsim.Moffat(beta=2.5, fwhm=seeing, trunc=seeing*4)

    os.makedirs(save_dir, exist_ok=True)

    # Creating fits file to save images
    fits = fitsio.FITS(
        os.path.join(save_dir, f'COSMOS_ngals={n_gals}_notnoisy.fits'),
        'rw',
        clobber=True
        )

    if batch_process == True:
        print("Batch processing images")
        with Pool(initializer=_init_worker,
                  initargs=(scale, psf, pixel_side, n_pixel_side, 
                            rng, var, zero_mag,
                            rand_flux, rand_size)) as pool:
            stamps = pool.starmap(generate_image, zip(galaxies, mags, hlr))
    else:
        raise ValueError("you should multriprocess")

    fits.write(None, header={}, extname='primary')

    for i, stamp in enumerate(stamps):
        fits.write(stamp.array, header={'IDENT': ids[i]},
                   extname=f'GALAXY_{i}')

    psf_image = psf.drawImage(nx=n_pixel_side, ny=n_pixel_side, scale=scale).array
    fits.write(psf_image, header={}, extname='PSF')

    fits.close()

    print(f"Simulated {n_gals} galaxies in {time() -t0:.2f} seconds")
    if args.noisy == 'True':
        noisy = True
    else:
        noisy = False
        print("not adding noise, finished generation.")


    if noisy:
        t1 = time()
        print(f"Now let's add some noise to the images")
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        ia_scene = []
        n_scenes = args.n_scenes
        gals_per_scene = n_gals // n_scenes
        noise_std = 0.37 * np.sqrt(10/args.sim_year)
        noise_variance = noise_std ** 2.0
        if args.force_detect=='True':
            do_force_detect = True
        else:
            do_force_detect = False
        buffer = args.buffer_size

        # create the new fits file
        fits = fitsio.FITS(
            os.path.join(save_dir, f'COSMOS_n_gals={n_gals}_noisy.fits'),
            'rw',
            clobber=True
        )
        n = np.sqrt(len(stamps))
        stamps_2 = stamps
        if n % 1 !=0:
            print("The number of stamps is not a square number. Padding with empty stamps")
            n = np.ceil(n).astype(int)
            n_empty = n**2 - len(stamps)

            for _ in range(n_empty):
                stamps.append(np.zeros((n_pixel_side, n_pixel_side)))
        else:
            n = int(n)

        with torch.no_grad():
            scene = torch.zeros((n*n_pixel_side, n*n_pixel_side), device=device)
            for k in range(n):
                for l in range(n):
                    try:
                        stamp_i = np.ascontiguousarray(stamps[k*n+l].array)
                    except AttributeError:
                        stamp_i = np.ascontiguousarray(stamps[k*n+l])
                    scene[k*n_pixel_side:(k+1)*n_pixel_side, l*n_pixel_side:(l+1)*n_pixel_side] = torch.tensor(stamp_i, device=device)

            if do_force_detect:
                pass
            else:
                scene = torch.nn.functional.pad(scene,
                                                (buffer, buffer, buffer, buffer),
                                                mode='constant', value=0)

            torch.manual_seed(args.seed)
            noise = torch.normal(mean=0.0, std=noise_std, size=scene.shape, device=device)
            scene=scene + noise

            torch.manual_seed(args.seed + 1e6)
            noise_array = torch.normal(mean=0.0, std=noise_std, size=scene.shape, device=device)
            del noise
            ia_scene.append(scene.cpu().numpy())
            del scene 

        torch.cuda.empty_cache()
        gc.collect()

        #now we split the scence back out into individual galaxy stamps:
        scene_nx = ia_scene[0].shape[0]
        scene_ny = ia_scene[0].shape[1]

        stamps_2 = []
        ncols = scene_nx // n_pixel_side
        for i in range(ncols):
            for j in range(ncols):
                stamp_i = ia_scene[0][i*n_pixel_side:(i+1)*n_pixel_side, j*n_pixel_side:(j+1)*n_pixel_side]
                try:
                    stamps[i*ncols+j].array = stamp_i
                except AttributeError:
                    pass

        fits.write(None, header={}, extname='primary')

        for i, stamp in enumerate(stamps):
            try:
                fits.write(stamp.array, header={'IDENT': ids[i]},
                           extname=f'GALAXY_{i}')
            except AttributeError:
                pass
        
        fits.close()
        t2 = time()
        print(f"added noise to galaxies, in {t2-t1:.2f} secs")
        print(f"total simulation and noisification took {t2-t0:.2f} secs")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate noisy LSST-like images from COSMOS galaxies')
    parser.add_argument('--n_gals', type=int, default=1,
                        help='Number of galaxies to simulate')
    parser.add_argument('--pixel_side', type=int, default=64,
                        help='Number of pixels on a side')
    parser.add_argument('--scale', type=float, default=0.2,
                        help='Pixel scale in arcseconds')
    parser.add_argument('--save_dir', type=str, default='.',
                        help='Directory to save images')
    parser.add_argument('--batch_process', type=str, default='True',
                        help='Batch process the images')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    parser.add_argument('--seeing', type=float, default=0.8, help='PSF FWHM')
    parser.add_argument('--var', type=float, default=0.05,
                        help='how much we change the flux or size by')
    parser.add_argument('--zero_mag', type=float, default=30,
                        help='Zeropoint magnitude')
    parser.add_argument('--random_sizeflux', type=str, default='both',
                        help='what should be randomized')
    parser.add_argument('--noisy', type=str, default='False',
                        help='Add noise to the images')
    parser.add_argument('--n_scenes', type=int, default=1,
                        help='number of scenes, only used in adding noise')
    parser.add_argument('--sim_year', type=int, default=1,
                        help='what LSST year are we simulating for noise deviation.')
    parser.add_argument('--force_detect', type=str, default='True', #Not sure if this one really is needed
                        help='Force to have a detection at the center of the image')
    parser.add_argument('--buffer_size', type=int, default=20, 
                        help='size of buffer to be added to noisy scene')
 
    args = parser.parse_args()

    main(args)
