import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from glob import glob


def gen_list():

    # Generate the testlist.dat for use with collapse2d and radialextract

    files = glob('CLEAR_OIIIcut/2D/*')
    f = open('testlist.dat', 'w')
    names, zz = np.loadtxt('best_ids_pos_z.dat', unpack=True, usecols=(0,3))

    dcheck = []
    for i,fil in enumerate(files):
        
        name = fil.split('_')
        name = name[2].split('.')
        id = int(name[0])

        check = np.where(names == id)
        if id in dcheck:
            continue
        if len(check[0]) > 0:
            f.write(str(id)+'  '+str(zz[check][0])+'  '+fil+'\n')

        dcheck.append(id)
        
    f.close()
    
    return


def collapse2d(infile='testlist.dat'):

    # Collapse the lines, radially, quadrant by quadrant

    outdir = 'collapse2d'
    stuff = np.genfromtxt(infile, dtype=None)

    ids, z, fils = [],[],[]
    for row in stuff:
        ids.append(row[0])
        z.append(row[1])
        fils.append(row[2])

    lines = [4861, 5007]

    for ii, f in enumerate(fils):
        print f

        hdu = fits.open(f)
        prim = hdu[0]
        fl2d = hdu[5].data
        err2d = hdu[6].data   # This is actually the 'WHT' frame, is that the same as error?
 
        ypix = fl2d.shape[0]
        center = int(np.round(ypix/2.))-2   # Will need to determine actual center. Is it always same place?
                                     # Maybe identify the row that has the brightest pixel? We'll see if necessary.
        wave_obs = hdu[9].data
        wave_rest = wave_obs/(1+z[ii])

        # So, this has to do with assigning weights of some kind. Not entirely certain what it's doing.
        err2d[np.where(err2d == 0)] = 10**np.max(err2d)
        ivar = 1/err2d**2
        ivar[np.where(err2d < np.median(err2d)/5)] = 1/(3*np.max(err2d))**2

        res = 0.13 # arcsec/pix - I think this is correct? True for WFC3, so I assume that's the scale for grism
        size = 1.2
        sizepix = int(np.round(size/res))
        
        # Not sure what sizepix is meant to be - I think a way of defining width of an emission line?
        if sizepix < center-1:
            sizepix = sizepix
        else:
            sizepix = center-1
        delta_wl = wave_rest[1] - wave_rest[0]

        # Subtract continuum
        # First create index array that masks where the lines are
        cont = fl2d * 0
        wlsize = 4 * sizepix * delta_wl
        nolines1 = np.logical_or((wave_rest < (lines[0]-2*wlsize)), (wave_rest > lines[0]+wlsize))
        nolines2 = np.logical_or((wave_rest < (lines[1]-wlsize)), (wave_rest > lines[1]+wlsize))
        nolines = np.logical_and(nolines1, nolines2)

        # Fit continuum
        for y in range(ypix):
            fit = np.polyfit(wave_rest[nolines], fl2d[y, nolines], 3)
            fit = np.flip(fit, 0)      # Why on earth does polyfit give the coefficient order from high->low??
            for i, p in enumerate(fit):
                cont[y,:] += p*wave_rest**i
                
        new2d = fl2d - cont
        newerr = err2d

        for i, l in enumerate(lines):
            if ((l-sizepix*delta_wl) < np.min(wave_rest)) or ((l+sizepix*delta_wl) > np.max(wave_rest)):
                continue
            linepix = np.where(np.abs(wave_rest-l) < delta_wl)
            linepix = linepix[0]
            if l == 5007:
                linepix[0] = (np.where(np.abs(wave_rest - 4861) < delta_wl))[0][0]
                linepix[1] = (np.where(np.abs(wave_rest - 5007) < delta_wl))[0][1]

        # Loop over 4 quadrants
        for quadrant in range(4):
            xtarg = 1 if quadrant >= 2 else 0
            xsign = -1 if xtarg == 0 else 1
            ysign = 1 if quadrant%2 == 0 else -1

            # Now, properly weight initial pixels in targe column - not sure what this means?
            # Ah...this is where ivar comes in. Hmm...
            totalweight = [0] * sizepix
            for  y in range(sizepix)[1:]:
                totalweight[y-1] = ivar[center+ysign*y, linepix[xtarg]]
                new2d[center+ysign*y, linepix[xtarg]] *= totalweight[y-1]
            numpixels = [1.] * sizepix

            # This chunk of code takes care of y = 0 column, divide between top and bottom halves
            frac = 0.5
            for x in range(sizepix)[1:]:
                weight = ivar[center, linepix[xtarg]+xsign*x]
                new2d[center+ysign*x, linepix[xtarg]] += weight * frac * new2d[center, linepix[xtarg]+xsign*x]
                totalweight[x-1] += frac * weight
                numpixels[x-1] += frac

                if ysign == -1:
                    new2d[center, linepix[xtarg]+xsign*x] = 0

            # Now do grid in x and y
            for x in range(sizepix)[1:]:
                for y in range(sizepix)[1:]:
                    
                    r = np.sqrt(x**2 + y **2)
                    rpixlo = int(np.round(r))
                    rpixhi = int(np.round(r+1))
                    fraclo = 1 - np.abs(r-rpixlo)
                    frachi = 1 - fraclo

                    if rpixlo > sizepix:
                        continue
                    if rpixhi > sizepix:
                        frachi = 0

                    weight = ivar[center+ysign*y, linepix[xtarg]+xsign*x]
                    new2d[center+ysign*rpixlo, linepix[xtarg]] += weight * fraclo * new2d[center+ysign*y, linepix[xtarg]+xsign*x]
                    totalweight[rpixlo-1] += fraclo * weight
                    numpixels[rpixlo-1] += fraclo

                    if frachi != 0:
                        new2d[center+ysign*rpixhi, linepix[xtarg]] += weight * frachi * new2d[center+ysign*y, linepix[xtarg]+xsign*x]
                        totalweight[rpixhi-1] += frachi * weight
                        numpixels[rpixhi-1] += frachi

                    new2d[center+ysign*y, linepix[xtarg]+xsign*x] -= (fraclo * new2d[center+ysign*y, linepix[xtarg]+xsign*x]
                                                                    + frachi * new2d[center+ysign*y, linepix[xtarg]+xsign*x])

            for y in range(sizepix)[1:]:      # normalize to total flux
                new2d[center+ysign*y, linepix[xtarg]] *= numpixels[y-1] / totalweight[y-1]
                newerr[center+ysign*y, linepix[xtarg]] = numpixels[y-1] / np.sqrt(totalweight[y-1])
               
        # Collapse 1D spectrum for check
        fl1dold = np.sum(fl2d[8:ypix-9,:], axis=0)
        err1dold = np.sum(err2d[8:ypix-9,:], axis=0)
        new1d = np.sum(new2d[8:ypix-9,:], axis=0)
        cont1d = np.sum(cont[8:ypix-9,:], axis=0)

        # Quality check on the continuum fitting
        #plt.step(wave_rest, fl1dold, 'k')
        #plt.step(wave_rest, new1d, 'r')
        #plt.step(wave_rest, cont1d, 'g')
        #plt.show()
        #plt.close()

        outdat = outdir+'/'+str(ids[ii])+'_spectrum.dat'
        f = open(outdat, 'w')
        for j, fl1 in enumerate(fl1dold):
            f.write(str(wave_rest[j])+'    '+str(fl1)+'    '+str(err1dold[j])+'/n')
        f.close()

        outfits = outdir+'/'+str(ids[ii])+'_collapse2d.fits'
        sci = fits.ImageHDU(data = new2d, name='SCI')
        err = fits.ImageHDU(data = newerr, name='ERR')
        wv = fits.ImageHDU(data = wave_rest, name='WAVE')
        hdulist = fits.HDUList([prim, sci, err, wv])
        hdulist.writeto(outfits, overwrite=True)
       
    return


def radial_extract(infile='testlist.dat'):

    # Divide 1D spectra in nuclear and extended components
    
    path = 'collapse2d'
    outdir = 'extract1d'

    # set how many pixels in each chunk
    inner = 3
    mid = 3
    outer = 6

    stuff = np.genfromtxt(infile, dtype=None)
    ids, z, fils = [],[],[]
    for row in stuff:
        ids.append(row[0])
        z.append(row[1])
        fils.append(row[2])

    hdu_sen = fits.open('WFC3.IR.G102.1st.sens.2.fits')
    sensit = hdu_sen[1].data['SENSITIVITY']
    wavele = hdu_sen[1].data['WAVELENGTH']

    for ii,f in enumerate(fils):
        print f

        # Read in 2D file
        file2d = path+'/'+str(ids[ii])+'_collapse2d.fits'
        hdu = fits.open(file2d)
        prim = hdu[0]
        fl2d = hdu[1].data
        err2d = hdu[2].data
        wl = hdu[3].data

        ivar = 1/err2d**2
        ivar[np.where(err2d == 0)] = 1/(10*np.max(err2d))**2

        ypix = fl2d.shape[0]
        center = int(np.round(ypix/2.))-2

        # Trim down sensitivity array to match size of grism
        idx = []
        for w in wl:
            idx.append((np.abs((wavele/(1+z[ii])) - w)).argmin())
        sens = sensit[idx]

        fl2d = fl2d/sens
        err2d = err2d/sens
        
        fracin = [0] * ypix
        fracout = [0] * ypix
        for i in xrange(center-(inner-1)/2, center+(inner-1)/2):
            fracin[i] = 1
        for i in xrange(center+mid, center+outer):
            fracout[i] = 1
        for i in xrange(center-outer, center-mid):
            fracout[i] = 1
        
        fl1din = [0] * fl2d[center,:]
        fl1dout = [0] * fl2d[center,:]

        fin = open(outdir+'/'+str(ids[ii])+'_rin.dat', 'w')
        fout = open(outdir+'/'+str(ids[ii])+'_rout.dat', 'w')

        for j in range(len(fl1din)):
            fl1din[j] = np.sum(fracin * fl2d[:,j] * ivar[:,j])/(np.sum(fracin * ivar[:,j])/np.sum(fracin))
            fl1dout[j] = np.sum(fracout * fl2d[:,j] * ivar[:,j])/(np.sum(fracout * ivar[:,j])/np.sum(fracout))
            errin = np.sqrt(np.sum(fracin*err2d[:,j]**2)/np.sum(fracin))
            errout = np.sqrt(np.sum(fracout*err2d[:,j]**2)/np.sum(fracout))
            fin.write(str(wl[j])+'    '+str(fl1din[j])+'    '+str(errin)+'\n')
            fout.write(str(wl[j])+'    '+str(fl1dout[j])+'    '+str(errout)+'\n')
            
        fin.close()
        fout.close()

        plt.step(wl, fl1dout, 'g-', label='Extended')
        plt.step(wl, fl1din, 'b-', label='Nuclear')
        plt.xlabel(r'Rest Wavelength ($\mathrm{\AA}$)')
        plt.ylabel(r'Flux (ergs s$^{-1}$ cm$^{-2}$ $\mathrm{\AA}^{-1}$)')
        plt.xlim(4700, 5200)
        plt.ylim(-1e-17, 1e-16)
        plt.legend()
        plt.savefig(outdir+'/'+str(ids[ii])+'_spec.pdf', dpi=300)
        plt.close()

    return
