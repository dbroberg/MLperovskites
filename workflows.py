"""For setting static perovskite"""

lpad_file_path = '/global/homes/d/dbroberg/atomate_fworkers/my_launchpad.yaml'

# import random
# import numpy as np
#
# from pymatgen.core import Element, Structure, Lattice
# from pymatgen.io.vasp import Poscar
#
# from atomate.vasp.fireworks.core import StaticFW
#
from fireworks import Workflow
from fireworks.core.launchpad import LaunchPad

from structure import PerfectPerovskite

from pymatgen.io.vasp.sets import MPRelaxSet
from atomate.vasp.fireworks.core import OptimizeFW

"""Lattice constant workflow related"""

def generate_lattconst_wf( list_elt_sets, functional='PBE', vasp_cmd = '>>vasp_cmd<<',
                           db_file = '>>db_file<<', submit=False):
    """Generates a workflow which calculates lattice constants through optimization fireworks for a given functional type"""

    if functional in ['PBE', 'LDA']:
        job_type = 'double_relaxation_run'
        potcar_type = functional
    elif functional in ['SCAN']:
        job_type = 'metagga_opt_run'
        potcar_type = 'PBE' #this is the POTCAR that needs to be used for SCAN...

    fws = []
    incar_settings = {"ADDGRID": True, 'EDIFF': 1e-8, 'NELMIN': 6}

    for elt_set in list_elt_sets:
        pp = PerfectPerovskite( Asite=elt_set[0], Bsite=elt_set[1], Osite=elt_set[2])
        s = pp.get_111_struct()

        vis = MPRelaxSet( s, user_incar_settings=incar_settings, potcar_functional=potcar_type)

        fw = OptimizeFW(s, name=s.composition.reduced_formula + " structure optimization",
                        vasp_input_set=vis, vasp_cmd=vasp_cmd, db_file=db_file,
                        job_type=job_type, auto_npar=">>auto_npar<<")
        fws.append( fw)

    wf = Workflow( fws, name='{} latt const workflow'.format( functional))
    if submit:
        print('Submitting workflow with {} fws for {}'.format( len(list_elt_sets), functional))
        lpad = LaunchPad().from_file(lpad_file_path)
        lpad.add_wf( wf)
    else:
        print('Workflow created with {} fws for {}'.format( len(list_elt_sets), functional))
        return wf

def parse_wf_for_latt_constants( wf_id):
    lpad = LaunchPad().from_file(lpad_file_path)

    wf = lpad.get_wf_by_fw_id( wf_id)

    lattdata = {}
    print('{}')


    return



"""Perturbation workflow related"""

class PerturbWFsetup(object):
    def __init__(self, perovskite, structure_type='111',
                 Nstruct = 100, perturbamnt=None,
                 vasp_cmd=">>vasp_cmd<<", db_file=">>db_file<<"):

        self.perovskite = perovskite

        self.Nstruct = Nstruct

        if perturbamnt is None:
            perturbamnt = perovskite.lattconst * 0.04
        self.perturbamnt = perturbamnt

        self.vasp_cmd = vasp_cmd
        self.db_file = db_file

        allowed_struct_type = ['111', '211', 's2s21', 's2s22']
        if structure_type not in allowed_struct_type:
            raise ValueError("{} not in {}".format( structure_type,
                                                    allowed_struct_type))

        self.structure_type = structure_type
        if self.structure_type == '111':
            self.base = perovskite.get_111_struct().copy()
        elif self.structure_type == '211':
            self.base = perovskite.get_211_struct().copy()
        elif self.structure_type == 's2s21':
            self.base = perovskite.get_sqrt2_1_struct().copy()
        elif self.structure_type == 's2s22':
            self.base = perovskite.get_sqrt2_2_struct().copy()


    # def _setup_perturbed_structs(self):
    #     self.track_N_structs = []
    #     for nind in range(self.Nstruct):
    #         thisstruct = self.sc_struct.copy()
    #         thisstruct.perturb( self.perturbamnt)
    #         self.track_N_structs.append( thisstruct)
    #
    #
    # def setup_wf(self, wfname=None):
    #     fws = []
    #
    #     incar_settings = {""} #TODO specify these...
    #
    #     short_name = self.base_struct.composition.pretty_formula
    #
    #     for struct_ind, struct in enumerate( self.track_N_structs):
    #         stat_fw = StaticFW(structure=struct.copy(),
    #                                name=short_name+'_Static_'+str(struct_ind),
    #                                vasp_input_settings = {"user_incar_settings": incar_settings},
    #                                vasp_cmd=self.vasp_cmd, db_file=self.db_file)
    #         fws.append( stat_fw.copy())
    #
    #     print('Setup {} workflow with {} fireworks'.format( short_name, self.Nstruct))
    #
    #     if wfname is None:
    #         wfname = 'PerovskiteWF_'+str(short_name)
    #
    #     wf = Workflow(fws, name=wfname)
    #
    #     return wf
    #
    # def submit_wf(self, lpad=None, wfname=None):
    #
    #     wf = self.setup_wf(wfname = wfname)
    #
    #     if lpad is None:
    #         lpad = LaunchPad().from_file(lpad_file_path)
    #
    #     lpad.add_wf( wf)


if __name__ == "__main__":
    init_list = [['Sr', 'Ti', 'O'],
                 ['Ca', 'Ti', 'O'],
                 ['Pb', 'Ti', 'O'],
                 ['Ba', 'Ti', 'O'],
                 ['Sn', 'Ti', 'O'],
                 ['Sr', 'Zr', 'O'],
                 ['Ca', 'Zr', 'O'],
                 ['Pb', 'Zr', 'O'],
                 ['Ba', 'Zr', 'O'],
                 ['Sn', 'Zr', 'O'],
                 ['Li', 'Nb', 'O'],
                 ['La', 'Al', 'O'],
                 ['Li', 'Ta', 'O'],
                 ['Y', 'Al', 'O'],
                 ['Gd', 'Sc', 'O'],
                 ['Dy', 'Sc', 'O'],
                 ['Nd', 'Sc', 'O'],
                 ['Sm', 'Sc', 'O'],
                 ['La', 'Lu', 'O']]

    for func in ['PBE', 'LDA', 'SCAN']:
        # generate_lattconst_wf(init_list, functional=func, submit=False)
        generate_lattconst_wf([init_list[0]], functional=func, submit=True)