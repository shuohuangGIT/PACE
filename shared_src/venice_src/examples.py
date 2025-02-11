import numpy as np
import matplotlib.pyplot as plt

from venice import Venice, dynamic_kick

from amuse.units import units, constants, nbody_system
from amuse.datamodel import Particles
from amuse.ic.plummer import new_plummer_model
from amuse.ic.brokenimf import MultiplePartIMF
from amuse.community.seba.interface import SeBa
from amuse.community.sse.interface import SSE
from amuse.community.ph4.interface import ph4


def gravity_stellar (gravity_code, stellar_code, converter, timestep, verbose=False):
    '''
    Create a Venice system coupling a gravity code to a stellar evolution code,
    where the mass change due to stellar evolution is propagated to gravity
    '''

    # Initialize gravity code
    gravity = gravity_code(converter)
    if hasattr(gravity.parameters, 'force_sync'):
        gravity.parameters.force_sync = True

    # Initialize stellar evolution code
    stellar = stellar_code()

    # Initialize Venice system
    system = Venice()
    system.verbose = verbose

    # Add codes
    system.add_code(gravity)
    system.add_code(stellar)

    # Add channel from code 1 (stellar) to code 0 (gravity), copying mass
    system.add_channel(1, 0, from_attributes=['mass'], to_attributes=['mass'])

    # Set coupling timestep between codes 0 and 1
    # Matrix is symmetric, so no need to do [1,0]
    system.timestep_matrix[0,1] = timestep

    return system, gravity, stellar


def bridge_to_potential (gravity_code, potential, converter, timestep,
        verbose=False):
    '''
    Create a Venice system coupling a gravity code to a background potential using a
    classic bridge
    '''

    # Initialize gravity code
    gravity = gravity_code(converter)
    if hasattr(gravity.parameters, 'force_sync'):
        gravity.parameters.force_sync = True

    # Initialize Venice system
    system = Venice()
    system.verbose = verbose

    # Add codes
    system.add_code(gravity)
    system.add_code(potential)

    # Add classic bridge kick from code 1 (potential) to code 0 (gravity)
    system.kick[1][0] = dynamic_kick

    # Set coupling timestep between codes 0 and 1
    # Matrix is symmetric, so no need to do [1,0]
    system.timestep_matrix[0,1] = timestep

    return system, gravity
