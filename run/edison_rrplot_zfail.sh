#!/bin/bash

# zfail 
#ipython -i -c '%run /global/homes/c/chahah/projects/packages/redrock/bin/rrplot --specfile="/global/cscratch1/sd/chahah/feasibgs/spectra/GamaLegacy.g15.expSpectra.darksky.seed1.exptime300.5of64blocks.zfail.fits" --rrfile="/global/cscratch1/sd/chahah/feasibgs/redrock/GamaLegacy.g15.expSpectra.darksky.seed1.exptime300.5of64blocks.zfail.redrock.h5"' --pylab

# zfail + archetype 
#ipython -i -c '%run /global/homes/c/chahah/projects/packages/redrock/bin/rrplot --specfile="/global/cscratch1/sd/chahah/feasibgs/spectra/GamaLegacy.g15.expSpectra.darksky.seed1.exptime300.5of64blocks.zfail.fits" --rrfile="/global/cscratch1/sd/chahah/feasibgs/redrock/GamaLegacy.g15.expSpectra.darksky.seed1.exptime300.5of64blocks.zfail.redrock.archetype.h5" --use-archetype --archetype="/global/homes/c/chahah/projects/packages/redrock-archetypes/"' --pylab

# zfail + archetype + new galaxy template
ipython -i -c '%run /global/homes/c/chahah/projects/packages/redrock/bin/rrplot --specfile="/global/cscratch1/sd/chahah/feasibgs/spectra/GamaLegacy.g15.expSpectra.darksky.seed1.exptime300.5of64blocks.zfail.fits" --rrfile="/global/cscratch1/sd/chahah/feasibgs/redrock/GamaLegacy.g15.expSpectra.darksky.seed1.exptime300.5of64blocks.zfail.redrock.archetype.newgaltemp.h5" --use-archetype --archetype="/global/homes/c/chahah/projects/packages/redrock-archetypes/" --templates="/global/cscratch1/sd/chahah/feasibgs/rrtemplate-galaxy.fits"' --pylab
