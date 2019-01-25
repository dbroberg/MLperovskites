"""For submitting workflows for perovskites"""

lpad_file_path = '/global/homes/d/dbroberg/atomate_fworkers/my_launchpad.yaml'

import os
import numpy as np

from pymatgen.core import Composition  #Element, Structure, Lattice
from pymatgen.io.vasp import Poscar, Outcar

# from atomate.vasp.fireworks.core import StaticFW
#
from fireworks import Workflow
from fireworks.core.launchpad import LaunchPad

from structure import PerfectPerovskite, StrainedPerovskite

from pymatgen.io.vasp.sets import MPRelaxSet

from atomate.vasp.fireworks.core import OptimizeFW
from atomate.vasp.workflows.base.ferroelectric import get_wf_ferroelectric

"""Lattice constant workflow related"""

def generate_lattconst_wf( list_elt_sets, functional='PBE', vasp_cmd = '>>vasp_cmd<<',
                           db_file = '>>db_file<<', submit=False, scan_smart_lattice=None):
    """Generates a workflow which calculates lattice constants
    through optimization fireworks for a given functional type

    NOTE: that the SCAN functionality might be reliant on some Custodian features from Danny's Custodian
    """

    if functional in ['PBE', 'LDA']:
        job_type = 'double_relaxation_run'
        potcar_type = functional
        incar_settings = {"ADDGRID": True, 'EDIFF': 1e-8}
    elif functional in ['SCAN']:
        job_type = 'metagga_opt_run'
        potcar_type = 'PBE' #this is the POTCAR that needs to be used for SCAN...
        incar_settings = {'EDIFF': 1e-8, 'ISIF': 7}
        if scan_smart_lattice is None:
            raise ValueError("Need to provide a smarter starting point "
                             "for SCAN lattice constants...")

    fws = []

    for elt_set in list_elt_sets:
        if functional == 'SCAN':
            compkey = Composition({elt_set[0]: 1, elt_set[1]: 1, elt_set[2]: 3})
            lattconst = scan_smart_lattice[compkey]
        else:
            lattconst = None
        pp = PerfectPerovskite( Asite=elt_set[0], Bsite=elt_set[1], Osite=elt_set[2],
                                lattconst = lattconst)
        s = pp.get_111_struct()

        vis = MPRelaxSet( s, user_incar_settings=incar_settings, potcar_functional=potcar_type)

        fw = OptimizeFW(s, name="{} {} structure optimization".format(s.composition.reduced_formula, functional),
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
    print('{} workflow retrieved with {} fws in it'.format( wf.name, len(wf.fws)))
    for fw in wf.fws:
        print('\t{}'.format( fw.name))
        if 'structure optimization' not in fw.name:
            raise ValueError("Not a recognized firework!")
        elif fw.state != 'COMPLETED':
            print('\t\tstatus = {}, so skipping'.format( fw.state))
            continue

        pat = fw.launches[-1].launch_dir
        s = Poscar.from_file( os.path.join( pat, 'CONTCAR.relax2.gz')).structure
        nom = s.composition.reduced_formula
        if nom in lattdata:
            raise ValueError("{} already exists in lattdata??".format( nom))
        elif (max(s.lattice.abc) - min(s.lattice.abc)) > 0.00001 or  (max(s.lattice.angles) - min(s.lattice.angles)) > 0.00001:
            raise ValueError("Error occured with lattice relaxation??".format( s.lattice))
        else:
            lattdata.update( {nom: s.lattice.abc[0]})

    print('\nFinalized lattice constant set:\n{}'.format( lattdata))

    return lattdata


"""LCALCEPs workflow related"""

def polarization_wf( polar_structure, nonpolar_structure, submit=False, wfid=None):

    if polar_structure.species != nonpolar_structure.species:
        raise ValueError("WRONG ORDER OF SPECIES: {} vs {}".format( polar_structure.species, nonpolar_structure.species))

    #TODO hack ability to add incar settings...
    # vasp_input_set_params = {'user_incar_settings': {"ADDGRID": True, 'EDIFF': 1e-8, "NELMIN": 6}}
    # wf = get_wf_ferroelectric( polar_structure, nonpolar_structure, vasp_cmd=">>vasp_cmd<<",
    #                           db_file='>>db_file<<', vasp_input_set_polar="MPStaticSet",
    #                           vasp_input_set_nonpolar="MPStaticSet", relax=False,
    #                           vasp_relax_input_set_polar=vasp_input_set_params,
    #                           vasp_relax_input_set_nonpolar=vasp_input_set_params,
    #                           nimages=5, hse=False, add_analysis_task=True,
    #                           wfid=wfid, tags=None)
    wf = get_wf_ferroelectric( polar_structure, nonpolar_structure, vasp_cmd=">>vasp_cmd<<",
                              db_file='>>db_file<<', relax=False,
                              nimages=9, hse=False, add_analysis_task=True,
                              wfid=wfid, tags=None)

    print('workflow created with {} fws'.format( len(wf.fws)))

    if submit:
        print("Submitting Polarization workflow")
        lp = LaunchPad().from_file( lpad_file_path)
        lp.add_wf(wf)
    else:
        return wf

def get_wf_timing( wf_id, returnval = False):

    lp = LaunchPad().from_file(lpad_file_path)
    wf = lp.get_wf_by_fw_id( wf_id)
    out_run_stats = []
    just_non_polar_stats = []
    for fw in wf.fws:
        ld = fw.launches[-1].launch_dir
        out = None
        if 'OUTCAR' in os.listdir(ld):
            out = Outcar( os.path.join( ld, 'OUTCAR'))
        elif 'OUTCAR.gz' in os.listdir(ld):
            out = Outcar( os.path.join( ld, 'OUTCAR.gz'))
        if out:
            out_run_stats.append( out.run_stats.copy())
            if 'nonpolar_polarization' in fw.name:
                just_non_polar_stats.append( out.run_stats.copy())
        ld += '/polarization'
        if os.path.exists( ld):
            out = None
            if 'OUTCAR' in os.listdir(ld):
                out = Outcar( os.path.join( ld, 'OUTCAR'))
            elif 'OUTCAR.gz' in os.listdir(ld):
                out = Outcar( os.path.join( ld, 'OUTCAR.gz'))
            if out:
                out_run_stats.append( out.run_stats.copy())
    cores = out_run_stats[0]['cores']
    print('Workflow {} retrieved {} Outcars all run on {} cores'.format( wf.name, len(out_run_stats), cores))
    timestat = {k: 0 for k in ['Elapsed time (sec)', 'System time (sec)',
                               'User time (sec)', 'Total CPU time used (sec)']}
    print('\nNon-Polar calc (non-polarization) alone took:')
    if len( just_non_polar_stats) != 1:
        raise ValueError("Too many non polar calcs?? = {}".format( len(just_non_polar_stats)))
    else:
        for k,v in just_non_polar_stats[0].items():
            if k in timestat.keys():
                print("\t{}: {}".format( k, round(v, 2)))

    for out in out_run_stats:
        if out['cores'] != cores:
            raise ValueError("Inconsisten number of cores for timing! {} vs {}".format( cores, out['cores']))
        for k,v in out.items():
            if k in timestat:
                timestat[k] += v

    print("\nSummary of TOTAL wf timing:")
    for k,v in timestat.items():
        print("\t{}: {}".format( k, round(v, 2)))
    if not returnval:
        return
    else:
        return {'tot': timestat, 'nonpolar': just_non_polar_stats}

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
                 ['La', 'Ta', 'O'],
                 ['Y', 'Al', 'O'],
                 ['Gd', 'Sc', 'O'],
                 ['Dy', 'Sc', 'O'],
                 ['Nd', 'Sc', 'O'],
                 ['Sm', 'Sc', 'O'],
                 ['La', 'Lu', 'O']]

    """Lattice constant generation related"""
    # for func in ['PBE', 'LDA']:
    #     generate_lattconst_wf(init_list, functional=func, submit=True)

    # gga_latt_dict = parse_wf_for_latt_constants( 3257)
    # generate_lattconst_wf([init_list[0]], functional='SCAN', submit=True,
    #                       scan_smart_lattice=gga_latt_dict)

    # gga_latt_dict = parse_wf_for_latt_constants( 3301)
    # lda_latt_dict = parse_wf_for_latt_constants( 3321)
    # from monty.serialization import dumpfn
    # from monty.json import MontyEncoder
    #
    # dumpfn( {'gga': gga_latt_dict, 'lda': lda_latt_dict}, 'latt_consts.json', cls=MontyEncoder)

    """Polarization testing related"""
    #first test on a cubic material (PbTiO3)
    from pymatgen import MPRester, Structure
    with MPRester() as mp:
        s = mp.get_structure_by_material_id('mp-19845')
    pert_coords = []
    for site in s.sites:
        if site.specie.symbol == 'Ti':
            pert_coords.append( site.coords + np.array( [0., 0., 0.25]))
        else:
            pert_coords.append( site.coords)
    pert_struct = Structure( s.lattice, s.species, pert_coords, coords_are_cartesian=True)

    # polarization_wf(s, pert_struct, submit=True, wfid="TestPbTiO3")
    polarization_wf(s, pert_struct, submit=True, wfid="BasicTessTestPbTiO3x2")

    # #second test on a tetragonal (known polar) material (PbTiO3)
    # #recreate this arxiv paper's result: https://arxiv.org/pdf/1702.04817.pdf
    # #teragonal cell:  a = 3.844 , c/a = 1.240,  volume = 70.4 , Displacement of Ti atom = 0.058 * c
    # #resulting polarization = 125.5 (μC/cm2)
    # from pymatgen.core import Lattice, Element, Structure
    # lattice = Lattice([[3.844, 0., 0.], [0., 3.844, 0.], [0., 0., 1.24 * 3.844]])
    # species = [Element("Pb"), Element("Ti"), Element("O"), Element("O"), Element("O")]
    # coords = [[0., 0., 0.], [0.5, 0.5, 0.5], [0.5, 0.5, 0.],
    #           [0.5, 0., 0.5], [0., 0.5, 0.5]]
    # perfect_struct = Structure( lattice, species, coords, coords_are_cartesian=False)
    #
    # pert_coords = perfect_struct.cart_coords[:]
    # pert_coords[1] += np.array([0., 0., 1.24 * 3.844 * 0.058])
    # pert_struct = Structure( lattice, species, pert_coords, coords_are_cartesian=True)
    #
    # polarization_wf(perfect_struct, pert_struct, submit=False, wfid="TestTetragonalPbTiO3")

    # #TRY above test again... with MPRester on
    # from pymatgen import MPRester, Structure
    # with MPRester() as mp:
    #     tmp_perfect_struct = mp.get_structure_by_material_id('mp-19845')
    #     pert_struct = mp.get_structure_by_material_id('mp-20459')
    # # reorder perfect structure sites as Ti, Pb, O, O, O
    # perfect_struct = Structure( tmp_perfect_struct.lattice,
    #                             [tmp_perfect_struct.species[ind] for ind in [3, 4, 0, 1, 2]],
    #                             [tmp_perfect_struct.cart_coords[ind] for ind in [3, 4, 0, 1, 2]],
    #                             coords_are_cartesian=True)
    # polarization_wf(perfect_struct, pert_struct, submit=False, wfid="Test3TetragonalPbTiO3")

    # # test on several randomly generated structures (cubic PbTiO3 ) to test timing
    # from pymatgen import MPRester
    # with MPRester() as mp:
    #     s = mp.get_structure_by_material_id('mp-19845')
    # bpc = PerfectPerovskite( Asite="Pb", Bsite="Ti", Osite="O", lattconst=s.lattice.abc[0])
    # for struct_type in ['111', '211', 's2s21', 's2s22']:
    #     print('\n-> Create workflow for {}'.format( struct_type))
    #     sp_class = StrainedPerovskite.generate_random_strain(bpc, structure_type=struct_type,
    #                                                          max_strain=0.06, perturb_amnt=None)
    #     polarization_wf( sp_class.base, sp_class.structure, submit = False)

    # get_wf_timing( 3809) #get timing for 111 PbTiO3 case

    #get timing for test distortions with 111, 211, s2s21, s2s22
    # outset = {}
    # for wf_id in [3823, 3830, 3837, 3844]:
    #     print("----> Doing {}".format(wf_id))
    #     out = get_wf_timing( wf_id, returnval=True)
    #     outset[wf_id] = out
    # print('\n------\n',outset)
