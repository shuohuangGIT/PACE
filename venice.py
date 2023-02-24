import numpy as np

from amuse.units import units

from symmetric_matrix import SymmetricMatrix


def find_connected_components (code_ids, dt, timestep_matrix):
    '''
    Find connected components among the timestep matrix of a set of codes

    code_ids: the indices of the codes to find connected components for
        (int array length (N))
    dt: pivot timestep of the connected component (scalar, units of time)
    timestep_matrix: matrix containing the target bridge timesteps 
        (array length (M,M), units of time, all entries in code_ids < M)
    '''

    N = len(code_ids)
    component_ids = -np.ones(N, dtype=int)

    counter = 0

    for x in range(N):

        if component_ids[x] < 0:

            component_ids[x] = counter

            stack = [x]

            while len(stack):

                i = stack.pop()
                I = code_ids[i]

                for j in range(N):

                    J = code_ids[j]

                    if component_ids[j] < 0 and timestep_matrix[I,J] < dt \
                            and timestep_matrix[I,J] > 0. | units.s:

                        component_ids[j] = counter
                        stack.append(j)

            counter += 1

    return component_ids


class NodeCodes:
    '''
    Node of a Venice connected component tree
    '''

    def __init__ (self, code_ids, dt, timestep_matrix):
        '''
        code_ids: the indices of the codes in this node (int array length (N))
        dt: pivot timestep of the node (scalar, units of time)
        timestep_matrix: matrix containing the target bridge timesteps 
            (array length (M,M), units of time, all entries in code_ids < M)

        Note that timestep_matrix could be reduced to (N,N) as only data for
        code_ids present is necessary; however, I am lazy.
        '''

        self.N_codes = len(code_ids)
        self.code_ids = code_ids

        self.dt = dt

        self.subdivide_node(timestep_matrix)


    def subdivide_node (self, timestep_matrix):
        '''
        Subdivide this node in connected components

        timestep_matrix: coupling timesteps of Venice system
        '''

        self.children = []

        if self.N_codes > 1:

            component_ids = find_connected_components(self.code_ids, self.dt,
                timestep_matrix)

            self.N_children = np.max(component_ids)+1

            for i in range(self.N_children):

                mask = component_ids == i

                self.children.append( NodeCodes(self.code_ids[ mask ], self.dt/2.,
                    timestep_matrix) )

        else:

            self.N_children = 0  


class Venice:

    def __init__ (self):

        self.codes = []

        self.timestep_matrix = None

        self.kick = None
        self.update_timestep = None
        self.sync_data = None
        self._channels = None

        self.model_time = 0. | units.s

        self.interlaced_drift = False
        self.verbose = False


    '''
    How to build Venice after initializing the class:

    Fundamentals
    - Add codes
        - using add_code(code)
    - Define coupling times
        - timestep_matrix[i,j] contains the maximum coupling timescale between codes
            i and j
        - the actual coupling timescale will be between 1 and 1/2 this value
        - if not defined (or set to <= 0), the coupling timescale is the total
            evolution time
        - timestep_matrix is automatically symmetric
    - Define channels
        - add_channel(i,j) adds a channel from a dataset of code i to one of code j
        - the default dataset is particles, but channels can be built between any
            Particles or Grid datasets
        - by default all properties are copied, but these can be set to any subset
        - add_iterable_channels and add_dynamic_iterable_channels function
            similarly, but for datasets that are iterated over (e.g. via the
            itergrids function in AMR codes). Note that add_iterable_channels adds
            a static set of channels, so is unsuited for codes with e.g. a dynamic
            set of grids. add_dynamic_iterable_grids makes channels when called,
            this is slower but can be used if the set of grids changes. 

    Bells & Whistles
    - Define kick functions
        - kick[i][j] describes an effect of code i on code j, which doesn't change
            code i
        - kick functions must be of the form kick_function(code i, code j,
            timestep)
        - each entry is a single function, multiple kick interactions must be
            aggregated to a single function
    - Define data synchronizers
        - sync_data[i][j] updates the structure of the dataset of code j with any
            changes made by code i
        - sync_data functions must be of the form sync_data(code i, code j)
        - this concerns e.g. added/removed particles and (de)refined grids
    - Define timestep updaters
        - update_timestep[i][j] computes the coupling timestep of codes i and j
          during evolution, must return a scalar with units of time
        - update_timestep functions must be of the form update_timestep(code i,
          code j, previous time step)
        - the tree will be updated to accomodate the new timesteps
    '''


    def evolve_model (self, end_time):
        '''
        Evolve the Venice system to an end time

        end_time: time to evolve to (scalar, units of time)
        '''

        dt = end_time - self.model_time

        self.root = NodeCodes(np.arange(len(self.codes)), dt, self.timestep_matrix)

        # Determine if dynamical timesteps are used (and cc tree needs recomputing)
        self._dynamic_timesteps = False
        for i in range(len(self.codes)):
          for j in range(len(self.codes)):
              if self.update_timestep[i][j] is not None:
                  self._dynamic_timesteps = True
                  break
          if self._dynamic_timesteps:
              break

        self._evolve_cc(self.root, dt, False)

        self.model_time = end_time


    def _evolve_cc (self, node, dt, reorganize):
        '''
        Evolve a node in the cc tree for a timestep

        node: cc tree node id to evolve (int)
        dt: timestep to evolve for (scalar, units of time)
        reorganize: flag to recompute tree structure from this node down (bool)
        '''

        # Only update tree if:
        #   a) this is after the second subcall
        #   b) there are dynamic timesteps (otherwise the tree is static)
        if reorganize and self._dynamic_timesteps:
            # UPDATE TIMESTEPS
            for code_id1 in node.code_ids:
              for code_id2 in node.code_ids:
                  if self.update_timestep[code_id1][code_id2] is not None:
                      self.timestep_matrix[code_id1, code_id2] = \
                          self.update_timestep[code_id1][code_id2](
                              self.codes[code_id1], self.codes[code_id2], dt)


            # UPDATE CC TREE
            node.subdivide_node(self.timestep_matrix)


        # RECURSIVE EVOLUTION
        for child1 in node.children:
            if child1.N_codes > 1:
                self._evolve_cc(child1, dt/2., False)
                for code_id in child1.code_ids:
                    for child2 in node.children:
                        if child1 != child2:
                            self._sync_code_to_codes(code_id, child2.code_ids)
                            self._copy_code_to_codes(code_id, child2.code_ids)


        # KICK
        for child1 in node.children:
          for child2 in node.children:
              if child1 != child2:
                  for code_id1 in child1.code_ids:
                    for code_id2 in child2.code_ids:
                        if self.kick[code_id1][code_id2] is not None:
                            self.kick[code_id1][code_id2](self.codes[code_id1],
                                self.codes[code_id2], dt/2.)
                            self._sync_code_to_codes(code_id2, node.code_ids)
                            self._copy_code_to_codes(code_id2, node.code_ids)


        # DRIFT
        single_codes = [ child.code_ids[0] for child in node.children \
            if child.N_codes == 1 and \
            hasattr(self.codes[child.code_ids[0]], 'evolve_model') ]

        if len(single_codes):
            if self.interlaced_drift:
                self._evolve_interlaced(single_codes, dt)
            else:
                self._evolve_linear(single_codes, dt)


        # KICK
        for child1 in node.children:
          for child2 in node.children:
              if child1 != child2:
                  for code_id1 in child1.code_ids:
                    for code_id2 in child2.code_ids:
                        if self.kick[code_id1][code_id2] is not None:
                            self.kick[code_id1][code_id2](self.codes[code_id1],
                                self.codes[code_id2], dt/2.)
                            self._sync_code_to_codes(code_id2, node.code_ids)
                            self._copy_code_to_codes(code_id2, node.code_ids)


        # RECURSIVE EVOLUTION
        for child1 in node.children:
            if child1.N_codes > 1:
                self._evolve_cc(child1, dt/2., True)
                for code_id in child1.code_ids:
                    for child2 in node.children:
                        if child1 != child2:
                            self._sync_code_to_codes(code_id, child2.code_ids)
                            self._copy_code_to_codes(code_id, child2.code_ids)


    def _evolve_interlaced (self, code_ids, dt):
        '''
        Evolve multiple codes with interlaced coupling for a timestep
        Codes are evolved using the general interlaced scheme, 
        with coupling in between

        code_ids: Venice code ids to evolve (list of ints)
        dt: timestep to evolve (scalar, units of time)
        '''

        N_codes = len(code_ids)

        if N_codes == 1:

            if self.verbose:
                print ("Evolving code {a} for {b} kyr".format(
                    a=code_id, b=dt.value_in(units.kyr)))

            self.codes[code_ids[0]].evolve_model( 
                self.codes[code_ids[0]].model_time + dt )

        else:

            code_ids1 = code_ids[:N_codes//2]
            code_ids2 = code_ids[N_codes//2:]

            self._evolve_interlaced(code_ids1, dt/2.)
            for code_id in code_ids1:
                self._sync_code_to_codes(code_id, code_ids2)
                self._copy_code_to_codes(code_id, code_ids2)

            self._evolve_interlaced(code_ids2, dt)
            for code_id in code_ids2:
                self._sync_code_to_codes(code_id, code_ids1)
                self._copy_code_to_codes(code_id, code_ids1)

            self._evolve_interlaced(code_ids1, dt/2.)
            for code_id in code_ids1:
                self._sync_code_to_codes(code_id, code_ids2)
                self._copy_code_to_codes(code_id, code_ids2)


    def _evolve_linear (self, code_ids, dt):
        '''
        Evolve multiple codes with linear coupling for a timestep
        Codes are evolved one by one for the full timestep, with coupling in between

        code_ids: Venice code ids to evolve (list of ints)
        dt: timestep to evolve (scalar, units of time)
        '''

        for code_id in code_ids:

            if self.verbose:
                print ("Evolving code {a} for {b} kyr".format(
                    a=code_id, b=dt.value_in(units.kyr)))

            self.codes[code_id].evolve_model( self.codes[code_id].model_time + dt )
            self._sync_code_to_codes(code_id, code_ids)
            self._copy_code_to_codes(code_id, code_ids)


    def _copy_code_to_codes (self, from_code_id, to_code_ids):
        '''
        Copy data from one code to a list of codes

        from_code_id: Venice code id to copy from (int)
        to_code_ids: Venice code ids to copy to (list of ints)
        '''

        for to_code_id in to_code_ids:
            for channel in self._channels[from_code_id][to_code_id]:
                if self.verbose:
                    print ("Copying from code {a} to code {b}".format(
                        a=from_code_id, b=to_code_id))
                channel.copy()


    def _sync_code_to_codes (self, from_code_id, to_code_ids):
        '''
        Synchronize data from one code to a list of codes

        from_code_id: Venice code id to copy from (int)
        to_code_ids: Venice code ids to copy to (list of ints)
        '''

        for to_code_id in to_code_ids:
            if self.sync_data[from_code_id][to_code_id] is not None:
                self.sync_data[from_code_id][to_code_id](
                    self.codes[from_code_id], self.codes[to_code_id])


    def add_channel (self, from_code_id, to_code_id, from_set_name='particles', 
            to_set_name='particles', from_attributes=None, to_attributes=None):
        '''
        Add a channel from one code's dataset to another code's dataset

        from_code: Venice code id of code to copy from (int)
        to_code: Venice code id of code to copy to (int)
        from_set_name: dataset to copy from (string)
        to_set_name: dataset to copy to (string)
        from_attributes: list of attributes to copy from code (list of strings)
        to_attributes: list of attributes to copy to code (list of strings)
        '''

        self._channels[from_code_id][to_code_id].append(
            getattr(self.codes[from_code_id], from_set_name).new_channel_to(
            getattr(self.codes[to_code_id], to_set_name),
            attributes=from_attributes, target_names=to_attributes)
        )


    def add_iterable_channels (self, from_code_id, to_code_id,
            from_iterator='itergrids', to_iterator='itergrids',
            from_attributes=None, to_attributes=None):
        '''
        Add channels from one code's iterable datasets to another code's iterable 
        datasets
        Use only if the set of grids doesn't vary, use add_dynamic_iterable_channels
        instead, but that method is slower!

        from_code: Venice code id of code to copy from (int)
        to_code: Venice code id of code to copy to (int)
        from_iterator: dataset iterator to copy from (string)
        to_iterator: dataset iterator to copy to (string)
        from_attributes: list of attributes to copy from code (list of strings)
        to_attributes: list of attributes to copy to code (list of strings)
        '''

        for from_grid, to_grid in zip(
                getattr(self.codes[from_code_id], from_iterator)(), 
                getattr(self.codes[to_code_id], to_iterator)()):

            self._channels[from_code_id][to_code_id].append(
                from_grid.new_channel_to(to_grid, 
                attributes=from_attributes, target_names=to_attributes)
            )


    def add_dynamic_iterable_channels (self, from_code_id, to_code_id,
            from_iterator='itergrids', to_iterator='itergrids',
            from_attributes=None, to_attributes=None):
        '''
        Add a dynamic channel generator and copier from one code's iterable
        datasets to another code's iterable datasets
        Use over add_iterable_channels when the set of iterable grids can vary,
        but this method is slower

        from_code: Venice code id of code to copy from (int)
        to_code: Venice code id of code to copy to (int)
        from_iterator: dataset iterator to copy from (string)
        to_iterator: dataset iterator to copy to (string)
        from_attributes: list of attributes to copy from code (list of strings)
        to_attributes: list of attributes to copy to code (list of strings)
        '''

        self._channels[from_code_id][to_code_id].append(
            DynamicIterableChannel(
                getattr(self.codes[from_code_id], from_iterator),
                getattr(self.codes[to_code_id], to_iterator),
                from_attributes=from_attributes, to_attributes=to_attributes)
            )


    def add_code (self, code):
        '''
        Add a code to Venice. See comments under __init__ for other
        initializations required to couple this code to the others.

        code: code to add to Venice
        '''

        N_codes = len(self.codes)


        self.codes.append(code)


        if N_codes > 0:
            for i in range(N_codes):
                self.kick[i].append(None)
                self.update_timestep[i].append(None)
                self.sync_data[i].append(None)
                self._channels[i].append([])

            self.kick.append([None]*(N_codes+1))
            self.update_timestep.append([None]*(N_codes+1))
            self.sync_data.append([None]*(N_codes+1))
            self._channels.append([[]]*(N_codes+1))


            new_timestep_matrix = SymmetricMatrix(N_codes+1, units.s)
            for i in range(N_codes):
                for j in range(i+1,N_codes):
                    new_timestep_matrix[i,j] = self.timestep_matrix[i,j]
            self.timestep_matrix = new_timestep_matrix

        else:
            self.kick = [[None]]
            self.update_timestep = [[None]]
            self.sync_data = [[None]]
            self._channels = [[[]]]

            self.timestep_matrix = SymmetricMatrix(1, units.s)


    def stop (self):
        '''
        Stop every code that has a stop function
        '''

        for code in self.codes:
            if hasattr(code, 'stop'):
                code.stop()


def dynamic_kick (kicker, kickee, dt):
    '''
    Kick function for classic dynamical kick. Lets a 'kicker' code provide velocity
    kicks on a 'kickee' code for a certain timestep.

    kicker: class that provides dynamical kicks. Must have a get_gravity_at_point
        function defined (class)
    kickee: class with particles that feel the dynamical kicks (class)
    dt: timestep to kick for (scalar, units of time)
    '''

    ax, ay, az = kicker.get_gravity_at_point(0. | kicker.particles.position.unit,
        kickee.particles.x, kickee.particles.y, kickee.particles.z)

    kickee.particles.vx += ax * dt
    kickee.particles.vy += ay * dt
    kickee.particles.vz += az * dt


class DynamicKick:

    def __init__ (self, radius_is_eps=False, h_smooth_is_eps=False,
            zero_smoothing=False):

        self.radius_is_eps = radius_is_eps
        self.h_smooth_is_eps = h_smooth_is_eps
        self.zero_smoothing = zero_smoothing


    def __call__ (self, kicker, kickee, dt):
        '''
        Kick function for classic dynamical kick. Lets a 'kicker' code provide 
        velocity kicks on a 'kickee' code for a certain timestep.

        kicker: class that provides dynamical kicks. Must have a 
        get_gravity_at_point function defined (class)
        kickee: class with particles that feel the dynamical kicks (class)
        dt: timestep to kick for (scalar, units of time)
        '''

        ax, ay, az = kicker.get_gravity_at_point(
            self._softening_lengths(kickee),
            kickee.particles.x, kickee.particles.y, kickee.particles.z)

        kickee.particles.vx += ax * dt
        kickee.particles.vy += ay * dt
        kickee.particles.vz += az * dt


    def _softening_lengths (self, code):
        if self.radius_is_eps:
            return code.particles.radius
        elif self.h_smooth_is_eps:
            return code.particles.h_smooth
        elif zero_smoothing:
            return 0.*code.particles.x
        elif hasattr(code, 'parameters') and hasattr(code.parameters, 
                'epsilon_squared'):
            return (code.parameters.epsilon_squared**0.5).as_vector_with_length(
                len(code.particles))
        else:
            return 0.*code.particles.x


class DynamicIterableChannel:
    '''
    Utility class for channels between codes that have a variable number of grids
    '''

    def __init__ (self, from_iterator, to_iterator,
            from_attributes=None, to_attributes=None):

        self.from_iterator = from_iterator
        self.to_iterator = to_iterator

        self.from_attributes = from_attributes
        self.to_attributes = to_attributes


    def copy (self):

        for from_grid, to_grid in zip(self.from_iterator(), self.to_iterator()):

            temp_channel = from_grid.new_channel_to(to_grid, 
                attributes=from_attributes, target_names=to_attributes)

            temp_channel.copy()
