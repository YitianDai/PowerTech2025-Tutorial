# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 17:48:33 2024

The code in Network.py provides the methods required to load and model
 a power system

@author: Eduardo Alejandro Martinez Cesena
https://www.researchgate.net/profile/Eduardo_Alejandro_Martinez_Cesena

@author: Yitian Dai
https://www.researchgate.net/profile/Yitian-Dai-2
"""

import numpy as np
import os
import pandapower as pp
import pandas as pd
import warnings


class Object(object):
    pass


class NetworkConfig:
    '''Default settings used for the Wind class'''

    def __init__(self):
        # Default time-step and map
        self.data = Object()

        # Location of network file
        self.data.path = os.path.join(os.path.dirname(__file__), '..', '..',
                                      'Inputs')
        self.data.name = 'case_ACTIVSg2000.xlsx'
        self.data.load = 'Historical_load.xlsx'
        self.data.DN = 'Distributed_generation.xlsx'
        self.data.line_status = 'LineStatus'

    def get_path(self):
        '''Get location of input file'''
        return self.data.path

    def get_name(self):
        '''Get name of file'''
        return self.data.name
    
    def get_load(self):
        '''Get historical load'''
        return self.data.load

    def set_path(self, val):
        '''Set location of input file'''
        self.data.path = val

    def set_name(self, val):
        '''Set name of file'''
        self.data.name = val

    def set_load(self, val):
        '''Set historical load'''
        self.data.load = val


class NetworkClass:
    def __init__(self, obj=None):
        '''Initialise Network class'''

        # Get default values
        if obj is None:
            obj = NetworkConfig()

        # Copy attributes
        for pars in obj.__dict__.keys():
            setattr(self, pars, getattr(obj, pars))

        # Load network data
        self._load_network()
        net = self._get_model()

        # Network information
        self.N = Object()
        self.N.num_lines = len(net.line)
        self.N.num_gen = len(net.sgen)
        self.N.num_buses = len(net.bus)
        self.N.num_loads = len(net.load)

    def _get_model(self):
        '''Get network model'''
        return self.model
    
    def _get_num_lines(self):
        '''Get number of lines'''
        return self.N.num_lines
    
    def _get_num_gen(self):
        '''Get number of generators'''
        return self.N.num_gen
    
    def _get_num_buses(self):
        '''Get number of buses'''
        return self.N.num_buses
    
    def _get_num_loads(self):
        '''Get number of loads'''
        return self.N.num_loads

    def _load_network(self):
        '''load network file'''
        # Bespoke method - assumes a pandapower file saved in excel format
        def fxn():
            full_path = os.path.join(self.data.path, self.data.name)
            model = pp.from_excel(full_path)
            return model

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            model = fxn()

        self._set_model(model)

    def _set_model(self, val):
        '''Set network model'''
        self.model = val

    def _set_num_lines(self, val):
        '''Set number of lines'''
        self.N.num_lines = val
    
    def _set_num_gen(self, val):
        '''Set number of generators'''
        self.N.num_gen = val
    
    def _set_num_buses(self, val):
        '''Set number of buses'''
        self.N.num_buses = val
    
    def _set_num_loads(self, val):
        '''Set number of loads'''
        self.N.num_loads = val

    def print(self):
        '''Print network'''
        # Bespoke moethod that relies of pandapower

    def _get_line_coords(self):
        '''Get the coordinates of the from-nodes and to-nodes of all lines'''
        net = self._get_model()
        
        all_coords = [pair for row in net.line_geodata['coords'] for pair in row]

        gis_line_bgn = all_coords[0::2]
        gis_line_end = all_coords[1::2]

        return gis_line_bgn, gis_line_end
        
    def _get_load_data(self):
        full_path = os.path.join(self.data.path, self.data.load)
        load_data = pd.read_excel(full_path, usecols=[1])
        return load_data
    
    def _get_load_hrly(self):
        temp_hist_load = self._get_load_data()
        net = self._get_model()
        sys_hrly_profile = \
            [(temp_hist_load.values[i*2][0]+temp_hist_load.values[i*2+1][0])/2
             for i in range(8760)]
        
        sys_hrly_profile /= np.max(sys_hrly_profile)

        num_load = self._get_num_loads()
        num_hrs_yr = len(sys_hrly_profile) 
        
        p_bus_hrly = np.ones((num_load, num_hrs_yr)) * net.load.loc[:, 'p_mw'].values[:, np.newaxis]
        q_bus_hrly = np.ones((num_load, num_hrs_yr)) * net.load.loc[:, 'q_mvar'].values[:, np.newaxis]

        # Adjust the loads based on the system hourly profile
        p_bus_hrly = p_bus_hrly * sys_hrly_profile
        q_bus_hrly = q_bus_hrly * sys_hrly_profile

        return p_bus_hrly, q_bus_hrly
    
    def _find_islands(self, net, output = None):
        import networkx as nx

        G = pp.topology.create_nxgraph(net)
        islands = list(nx.connected_components(G))

        if output:
            print(f"Number of islands: {len(islands)}")
            for i, island in enumerate(islands, start=1):
                print(f"Island {i}: {island}")
        
        return islands
    
    def _MPC_by_Parts(self, net): 
        """
        Run DC OPF for each islanded part of the network independently.
        Returns combined results for all islands.
        """
        net_opf = net.deepcopy()
        results = net.deepcopy()
        islands = self._find_islands(net_opf)

        for key in ['res_bus', 'res_line', 'res_trafo', 'res_sgen', 'res_ext_grid', 'res_load']:
            if hasattr(results, key):
                setattr(results, key, getattr(results, key).iloc[0:0])  # clear the table safely

        for i, island in enumerate(islands):
            npc = net_opf.deepcopy()

            bus_indices_to_keep = set(island)
            all_bus_indices = set(npc.bus.index)
            buses_to_remove = list(all_bus_indices - bus_indices_to_keep)
            pp.drop_buses(npc, buses_to_remove)
            if len(npc.load) == 0:
                continue
   
            if len(npc.ext_grid) == 0:
                slack_bus_idx = npc.sgen.iloc[0]['bus']
                pp.create_ext_grid(npc, bus=slack_bus_idx, vm_pu=1.0, name="Slack Bus", max_p_mw=npc.sgen.iloc[0]['max_p_mw'])

            try:
                pp.rundcopp(npc)
                for res_key in ['res_bus', 'res_line', 'res_trafo', 'res_sgen', 'res_ext_grid', 'res_load']:
                    if hasattr(npc, res_key):
                        npc_result = getattr(npc, res_key)
                        if not npc_result.empty:
                            if getattr(results, res_key).empty:
                                setattr(results, res_key, npc_result.copy())
                            else:
                                updated = pd.concat([getattr(results, res_key), npc_result])
                                setattr(results, res_key, updated)
            except Exception as e:
                # Fail-safe: set all static generators to max output
                for i, sgen in npc.sgen.iterrows():
                    bus_idx = sgen['bus']
                    p_max = sgen.get('max_p_mw', sgen['p_mw'])
                    res_row = pd.DataFrame([[bus_idx, p_max]], columns=['bus', 'p_mw'], index=[i])
                    results.res_sgen = pd.concat([results.res_sgen, res_row])

        return results

    def _add_virtual_gen(self, net):
        """
        Add virtual generators to each load bus for load shedding simulation.
        Each virtual generator is assigned a high cost to ensure it is only used as a last resort.
        """
        net1 = net.deepcopy()
        load_bus = net1.load.iloc[:]['bus']
        NoGen = len(net1.sgen)

        for i in range(len(load_bus)):

            # Create dummy generators
            pp.create_sgen(net1, load_bus[i], p_mw=0, min_p_mw=0,
                                    max_p_mw=net.load.p_mw[i], controllable=True, slack=False)
            pp.create_poly_cost(net1, NoGen+i, et='sgen',
                                cp1_eur_per_mw=1000,cp2_eur_per_mw2=1)

        try:
            pp.rundcopp(net1)
            converged = net1.OPF_converged
            print(f"\nOPF calculation complete. Converged: {converged}")
            if converged:
                print("✅ DC OPF converged successfully with virtual generators.")
            else:
                print("❌ DC OPF did not converge. Please check the network configuration.")
        except Exception as e:
            print(f"❌ DC OPF failed to converge. Error: {e}")
        return net1
    
    def _load_line_status(self, filename):
        full_path = os.path.join(self.data.path, self.data.line_status, filename)
        Line_status = pd.read_excel(full_path)
        return Line_status
    
    def _get_DN_data(self):
        full_path = os.path.join(self.data.path, self.data.DN)
        load_data = pd.read_excel(full_path, usecols="B:C")
        return load_data
    
    def _get_flex_hrly(self, Num_DG_bus):
        import random

        temp_DN = self._get_DN_data()
        net = self._get_model()

        DN_profile = [temp_DN.values[i][0] for i in range(np.shape(temp_DN)[0])]

        num_DG_bus = Num_DG_bus
        num_load = len(net.load)
        num_hrs_yr = 8760

        DG_bus = random.sample(list(net.load.index), round(num_DG_bus*num_load))
        p_bus_hrly = np.ones((num_load,  num_hrs_yr)) * net.load.loc[:, 'p_mw'].values[:, np.newaxis]

        for i in range(len(DN_profile)):
            p_bus_hrly[DG_bus, i] = p_bus_hrly[DG_bus, i] * DN_profile[i]
        
        return p_bus_hrly

    def _get_essential_demand_hrly(self):
        temp_DN = self._get_DN_data()
        net = self._get_model()

        DN_profile = [temp_DN.values[i][1] for i in range(np.shape(temp_DN)[0])]

        num_load = len(net.load)
        num_hrs_yr = 8760
        p_bus_hrly = np.ones((num_load,  num_hrs_yr)) * net.load.loc[:, 'p_mw'].values[:, np.newaxis]

        for i in range(len(DN_profile)):
            p_bus_hrly[:, i] = p_bus_hrly[:, i] * DN_profile[i]
        
        return p_bus_hrly

