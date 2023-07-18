
from darts.engines import sim_params, value_vector
from physics.physics_comp_sup import SuperPhysics
import numpy as np
from physics.properties_basic import *
from physics.property_container import *
from select_para import props
from darts.tools.keyword_file_tools import load_single_keyword
from darts.models.reservoirs.struct_reservoir import StructReservoir
from darts.models.darts_model import DartsModel
import numpy as np
from darts.tools.keyword_file_tools import load_single_keyword, save_few_keywords
import os


# Model class creation here!
class Model(DartsModel):
    def __init__(self, perm, por, wells_config, n_points=1000):
        # call base class constructor
        super().__init__()
        self.n_points = n_points
        # measure time spend on reading/initialization      
        self.timer.node["initialization"].start()

        self.wells_config = wells_config

        
        self.nx = 32
        self.ny = 32
        nz = 1

        perm = perm.values.flatten()
        por = por.values.flatten()
     
        self.permx = perm
        self.permy = perm
        self.permz = perm*0.1
        self.poro = por

        # if size  == 256:
        #     dx = 24
        # elif size  == 128:
        #     dx = 48
        # elif size  == 64:
        #     dx = 96
        # elif size  == 32:
        #     dx = 192
        # elif size  == 16:
        #     dx = 384
        # elif size  == 8:
        #     dx = 768


        self.dx = 192
        self.dy = 192
        self.dz = 10
        #self.depth = depth_data

        # Import other properties from files
        self.actnum = 1
        

        #tranx = load_single_keyword('eclipse/TRANS.in', 'TRANX')
        #trany = load_single_keyword('eclipse/TRANS.in', 'TRANY')
        #tranz = load_single_keyword('eclipse/TRANS.in', 'TRANZ')
        #self.calc_tpfa_transmissibilities(nx, ny, nz, nres)

        self.reservoir = StructReservoir(self.timer, nx=self.nx, ny=self.ny, nz=nz, dx=self.dx, dy=self.dy, dz=self.dz,
                                         permx=self.permx, permy=self.permy, permz=self.permz, poro=self.poro,
                                         depth=2000, actnum=self.actnum)

        
   


        well_dia = 0.152
        well_rad = well_dia / 2

        self.reservoir.inj_wells = []
        self.reservoir.prod_wells = []

        # Add wells from wells_config
        for well_name in wells_config.WellName.values:
            well_row = wells_config.sel(WellName=well_name)

            self.reservoir.add_well(well_name, wellbore_diameter=well_dia)
            self.reservoir.add_perforation(self.reservoir.wells[-1], well_row['i'].values, well_row['j'].values, 1,
                                        well_radius=well_rad, multi_segment=False, verbose=1)
            
            # Set wells as injectors or producers based on 'WellType' attribute in the Dataset
            if well_row['WellType'].values == 'INJECTOR':
                self.reservoir.inj_wells.append(self.reservoir.wells[-1])
            elif well_row['WellType'].values == 'PRODUCER':
                self.reservoir.prod_wells.append(self.reservoir.wells[-1])

        self.timer.node["initialization"].stop()



  

        self.zero = 1e-8
        """Physical properties"""
        # Create property containers:
        components_name = ['CO2', 'C1', 'H2O']
        Mw = [44.01, 16.04, 18.015]
        for name in components_name:
            Mw.append(props(name, 'Mw'))
        self.property_container = property_container(phases_name=['gas', 'wat'],
                                                     components_name=components_name,
                                                     Mw=Mw, min_z=self.zero / 10)
        self.components = self.property_container.components_name
        self.phases = self.property_container.phases_name

        """ properties correlations """
        self.property_container.flash_ev = Flash(self.components, [4, 2, 0.01], self.zero)
        self.property_container.density_ev = dict([('gas', Density(compr=1e-3, dens0=200)),
                                                   ('wat', Density(compr=1e-5, dens0=600))])
        self.property_container.viscosity_ev = dict([('gas', ViscosityConst(0.015)),
                                                     ('wat', ViscosityConst(0.5))])
        self.property_container.rel_perm_ev = dict([('gas', PhaseRelPerm("gas")),
                                                    ('wat', PhaseRelPerm("oil"))])

        """ Activate physics """
        self.physics = SuperPhysics(self.property_container, self.timer, n_points=200, min_p=1, max_p=1000,
                                    min_z=self.zero/10, max_z=1-self.zero/10)

        self.inj_stream = [1.0 - 2 * self.zero, self.zero]
        self.ini_stream = [self.zero, 0.8]

        # Some newton parameters for non-linear solution:
        self.params.first_ts = 1e-2
        self.params.mult_ts = 2
        self.params.max_ts = 10

        self.params.tolerance_newton = 1e-2
        self.params.tolerance_linear = 1e-3
        self.params.max_i_newton = 50#20
        self.params.max_i_linear = 60#30
        self.params.newton_type = sim_params.newton_local_chop

    # Initialize reservoir and set boundary conditions:
    def set_initial_conditions(self):
        """ initialize conditions for all scenarios"""
        self.physics.set_uniform_initial_conditions(self.reservoir.mesh, 200, self.ini_stream)

    def set_op_list(self):
        self.op_num = np.array(self.reservoir.mesh.op_num, copy=False)
        n_res = self.reservoir.mesh.n_res_blocks
        self.op_num[n_res:] = 1
        self.op_list = [self.physics.acc_flux_itor, self.physics.acc_flux_w_itor]

    def export_pro_vtk(self, file_name='Realization'):
        Xn = np.array(self.physics.engine.X, copy=False)
        P = Xn[0:self.reservoir.nb * 3:3]
        z1 = Xn[1:self.reservoir.nb * 3:3]
        z2 = Xn[2:self.reservoir.nb * 3:3]
        z3 = 1 - z1 - z2
        sg = np.zeros(len(P))
        sw = np.zeros(len(P))

        for i in range(len(P)):
            values = value_vector([0] * self.physics.n_ops)
            state = value_vector((P[i], z1[i], z2[i]))
            self.physics.property_itor.evaluate(state, values)
            sg[i] = values[0]
            sw[i] = 1 - sg[i]

        self.export_vtk(file_name, local_cell_data={'CO_2':z1, 'C_1': z2, 'H2O': z3, 'GasSat': sg, 'WatSat': sw, 'Pressure': P})
    
    def set_wells(self, step):
        gas_rates = self.wells_config.sel(steps=step)['gas_rate']    
        
        for i, w in enumerate(self.reservoir.wells): 
            gas_rate = gas_rates.sel(WellName=w.name).values
            w.control = self.physics.new_rate_inj(gas_rate, self.inj_stream, 0)
                    
            #w.control = self.physics.new_bhp_inj(600, self.inj_stream) #approximatly the litostatic pressure (1psi/ft)

            #    w.control = self.physics.new_bhp_inj(400, self.inj_stream)
            #else:
            #    w.control = self.physics.new_bhp_prod(100)
    

    def set_boundary_conditions(self):
        for i, w in enumerate(self.reservoir.wells):
            w.control= self.physics.new_rate_inj(0, self.inj_stream, 0)
            

            
#save xarray as netcdf extension

def save_xarray_as_netcdf(xarray, file_name):
    import xarray as xr
    xr.save_mfdataset(xarray, file_name)
