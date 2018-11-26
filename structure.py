"""For setting static perovskite"""

lpad_file_path = '/global/homes/d/dbroberg/atomate_fworkers/my_launchpad.yaml'

import random
import numpy as np

from pymatgen.core import Element, Structure, Lattice

allowed_struct_type = ['111', '211', 's2s21', 's2s22']


class PerfectPerovskite(object):

    def __init__(self, Asite="Pb", Bsite="Ti", Osite="O",
                 lattconst=None):
        self.eltA = Element(Asite)
        self.eltB = Element(Bsite)
        self.eltC = Element(Osite)
        if lattconst is None: #guess lattice constant based on perfect perovskite tol factor
            try:
                lattconst = (self.eltB.ionic_radii[4] + self.eltC.ionic_radii[-2]) * 2.
            except:
                lattconst = (max(self.eltB.ionic_radii.values()) +
                             max(self.eltC.ionic_radii.values())) * 2.
            lattconst = float(lattconst)
            print('No lattice constant given. Using {} for {}{}{}3'.format( lattconst, Asite,
                                                                              Bsite, Osite))

        self.lattconst = lattconst

    def get_111_struct(self):
        lattice = Lattice( self.lattconst * np.identity(3))
        species = [self.eltA, self.eltB, self.eltC,
                   self.eltC, self.eltC]
        coords = [[0., 0., 0.], [0.5, 0.5, 0.5], [0.5, 0.5, 0.],
                  [0.5, 0., 0.5], [0., 0.5, 0.5]]

        struct = Structure(lattice, species, coords,
                                coords_are_cartesian=False)
        return struct.copy()

    def get_211_struct(self):
        struct = self.get_111_struct()
        struct.make_supercell( [2, 1, 1])

        return struct.copy()

    def get_sqrt2_1_struct(self):
        """Setup a sqrt 2 x sqrt 2 x 1 structure"""

        orig_scaled_mat = np.array([[self.lattconst * np.sqrt(2), 0., 0.],
                                    [0., self.lattconst * np.sqrt(2), 0.],
                                    [0., 0., self.lattconst]])
        rotation = np.array([[np.cos(45. * np.pi / 180.), -np.sin(45. * np.pi / 180.), 0],
                             [np.sin(45. * np.pi / 180.), np.cos(45. * np.pi / 180.), 0],
                             [0., 0., 1.]])
        rotated_mat = np.dot(orig_scaled_mat, rotation)
        lattice = Lattice(rotated_mat)
        species = [self.eltA, self.eltA,
                   self.eltB, self.eltB,
                   self.eltC, self.eltC, self.eltC,
                   self.eltC, self.eltC, self.eltC]
        coords = [[0.5, 0., 0.5], [0., 0.5, 0.5],
                  [0., 0., 0.], [0.5, 0.5, 0.],
                  [0.25, 0.25, 0.], [0.75, 0.75, 0.], [0.25, 0.75, 0.],
                  [0.75, 0.25, 0.], [0., 0., 0.5], [0.5, 0.5, 0.5]]

        struct = Structure(lattice, species, coords,
                           coords_are_cartesian=False)

        return struct.copy()

    def get_sqrt2_2_struct(self):
        """Setup a sqrt 2 x sqrt 2 x 2 structure"""
        struct = self.get_sqrt2_1_struct()
        struct.make_supercell( [1, 1, 2])
        return struct.copy()

    def get_struct_from_structure_type(self, structure_type='111'):

        if structure_type not in allowed_struct_type:
            raise ValueError("{} not in {}".format( structure_type,
                                                    allowed_struct_type))
        if structure_type == '111':
            base = self.get_111_struct()
        elif structure_type == '211':
            base = self.get_211_struct()
        elif structure_type == 's2s21':
            base = self.get_sqrt2_1_struct()
        elif structure_type == 's2s22':
            base = self.get_sqrt2_2_struct()

        return base




class StrainedPerovskite( object):

    def __init__(self, base_perovskite_cls, strain_tensor, atomic_perturbations,
                 structure_type='111'):
        self.base_perovskite_cls = base_perovskite_cls

        if structure_type not in allowed_struct_type:
            raise ValueError("{} not in {}".format( structure_type,
                                                    allowed_struct_type))
        self.structure_type = structure_type
        self.base = base_perovskite_cls.get_struct_from_structure_type( structure_type)

        self.strain_tensor = strain_tensor

        if len( atomic_perturbations) != len( self.base):
            raise ValueError("number of perturbations given = {} but there are {} "
                             "atoms...".format( len(atomic_perturbations), len(self.base)))

        self.atomic_perturbations = atomic_perturbations

    @property
    def structure(self):
        strained_lattice = perform_strain( self.base.lattice.matrix, self.strain_tensor)
        species, coords = [], []
        for ind, perturb in enumerate(self.atomic_perturbations):
            species.append( self.base.species[ind])
            coords.append( self.base.cart_coords[ind] + np.array(perturb) )

        struct = Structure(strained_lattice, species, coords,
                           coords_are_cartesian=True)

        return struct.copy()

    @staticmethod
    def generate_random_strain( base_perovskite_cls, structure_type = '111',
                                    max_strain = 0.06, perturb_amnt = None):

        strain_tensor = random_strain( max_strain)

        if perturb_amnt is None:
            perturb_amnt = base_perovskite_cls.lattconst * 0.04

        if structure_type not in allowed_struct_type:
            raise ValueError("{} not in {}".format( structure_type,
                                                    allowed_struct_type))
        def get_rand_vec():
            vector = np.random.randn(3)
            vnorm = np.linalg.norm(vector)
            return vector / vnorm * perturb_amnt if vnorm != 0 else get_rand_vec()

        base_struct = base_perovskite_cls.get_struct_from_structure_type( structure_type)
        for siteind, site in enumerate(base_struct.sites): #find the first A-site atom, because we will not perturb this atom
            if site.specie == base_perovskite_cls.eltA:
                break
        atomic_perturbations = [get_rand_vec() if ind != siteind else np.array([0.,0.,0.]) for ind in range(len(base_struct))]

        return StrainedPerovskite( base_perovskite_cls, strain_tensor,
                                   atomic_perturbations, structure_type=structure_type)

    @staticmethod
    def generate_from_final_structures( perfect_structure, strained_structure,
                                        guess_B_site = None):
        """
        From two structures
        :param perfect_structure:
        :param strained_structure:
        :return:
        """
        if len(perfect_structure) != len(strained_structure):
            raise ValueError("Inconsistent number of atoms!!")

        abc = perfect_structure.lattice.abc
        if len(perfect_structure) == 5:
            structure_type = '111'
        elif len(perfect_structure) == 20:
            structure_type = 's2s22'
        elif abs( 1. - abc[1] / abc[2]) < 0.1:
            structure_type = '211'
        else:
            structure_type = 's2s21'

        # generate base perovskite class
        if structure_type == 's2s22':
            latt_const = abc[2]/2.
        else:
            latt_const = abc[2]

        red_comp = perfect_structure.composition.reduced_composition
        assume_B_site = ["Ti", "Zr", "Nb", "Al", "Ta", "Sc", "Lu"]
        if guess_B_site:
            assume_B_site.extend([guess_B_site])

        for k,v in red_comp.items():
            if v == 3.:
                Osite = k.symbol
            elif k.symbol in assume_B_site:
                Bsite = k.symbol
            else:
                Asite = k.symbol

        base_pv = PerfectPerovskite(Asite=Asite, Bsite=Bsite,
                                    Osite=Osite, lattconst=latt_const)

        # get perturbation list (with periodic boundary conditions and accounting for different lattices)
        atomic_perturbations = []
        for site_init, site_final in zip(perfect_structure, strained_structure):
            frac_final_site_in_perf_lattice = perfect_structure.lattice.get_fractional_coords( site_final.coords)
            dist, jimage = site_init.distance_and_image_from_frac_coords(frac_final_site_in_perf_lattice)
            vec_init_to_final = perfect_structure.lattice.get_cartesian_coords(frac_final_site_in_perf_lattice -
                                                                                jimage - site_init.frac_coords)
            atomic_perturbations.append( vec_init_to_final[:])

        # get lattice strain
        original_lattice_matrix = perfect_structure.lattice.matrix
        strained_lattice = strained_structure.lattice.matrix
        strain_tensor = np.dot( np.linalg.pinv( original_lattice_matrix),
                                strained_lattice)
        strain_tensor = strain_tensor.T

        confirm_strained_lattice = perform_strain( original_lattice_matrix,  strain_tensor)
        if Lattice(confirm_strained_lattice) != Lattice(strained_lattice):
            raise ValueError("Problem matching strain. Generated lattice:\n{}\n"
                             "Correct Strained Lattice:\n{}".format( confirm_strained_lattice,
                                                                     strained_lattice))

        return StrainedPerovskite( base_pv, strain_tensor,
                                   atomic_perturbations, structure_type=structure_type)



def perform_strain( original_lattice_matrix,  strain_tensor):
    """
    Strains self (in place) by the tensor strain_tensor.
    The strain tensor should take the form of a deformation gradiation tensor F (doesn't have to be symmetric). See examples below.

    Argument strain_tensor can be a 3x3 tensor or a 1x6 Voigt-notated tensor

    Note: 1.0 is not automatically added to the diagonal components of the strain tensor.

                            | e11 e12 e13 |
    full tensor looks like:	| e21 e22 e23 |      where e12 = (dx_1/dX_2), the constant factor by which the x component of a vector (in new coordinates) is displaced per y component in original coordinates.
                            | e31 e32 e33 |      e12 would then correspond to a shear strain applied on the y-plane in the x direction.

    voigt equivalent is: (e11, e22, e33, 2*e23, 2*e13, 2*e12)

    For an original vector a, the strain tensor will take it to a new vector a' given by:

    a'_x = a_x*e11 + a_y*e12 + a_z*e13
    a'_y = a_x*e21 + a_y*e22 + a_z*e23
    a'_z = a_x*e31 + a_y*e32 + a_z*e33

    Another example: if e12 (a shear applied to the y-plane in the x direction) is 0.1, and if b was originally [0.0, 10.0, 0.0], the new b vector after shearing will be [1.0, 10.0, 0.0]

    If the strain tensor is the identity matrix, the lattice will not change.

    For out of plane only, specify e13, e23, and e33 as non-zero.
    """
    original_lattice_matrix = np.array( original_lattice_matrix)
    strain_tensor = np.array(strain_tensor)

    if strain_tensor.ndim == 1:
        strain_tensor = convert_voigt_strain_to_3x3_tensor(strain_tensor)

    strained_lattice = np.dot(original_lattice_matrix, strain_tensor.T)

    return strained_lattice


def convert_voigt_strain_to_3x3_tensor(voigt_tensor):
    """
    Converts 1x6 like [e1, e2, e3, e4, e5, e6] to 3x3 like [[e1, e6/2, e5/2], [e6/2, e2, e4/2], [e5/2, e4/2, e3]] (as a numpy array)
    """

    voigt_tensor = np.array(voigt_tensor)

    if not voigt_tensor.ndim == 1:
        raise Exception("Number of array dimensions of voigt tensor must be 1. Input tensor: " + str(voigt_tensor))

    if not len(voigt_tensor) == 6:
        raise Exception("Voigt tensor must have six components. Input tensor: " + str(voigt_tensor))

    full_tensor = []

    full_tensor.append([voigt_tensor[0], voigt_tensor[5] / 2.0, voigt_tensor[4] / 2.0])
    full_tensor.append([voigt_tensor[5] / 2.0, voigt_tensor[1], voigt_tensor[3] / 2.0])
    full_tensor.append([voigt_tensor[4] / 2.0, voigt_tensor[3] / 2.0, voigt_tensor[2]])

    return np.array(full_tensor)


def random_strain( max_strain):
    def r():
        return random.uniform(-1.0 * max_strain, max_strain)

    strain_tensor = [[1.0 + r(), r() / 2.0, r() / 2.0],
                     [r() / 2.0, 1.0 + r(), r() / 2.0],
                     [r() / 2.0, r() / 2.0, 1.0 + r()]]
    return strain_tensor



def print_all_types(straining = False):
    from pymatgen.io.vasp import Poscar
    sclass = PerfectPerovskite()

    for stype in allowed_struct_type:

        if straining:
            strained = StrainedPerovskite.generate_random_strain(sclass, structure_type=stype,
                                                                 max_strain=0.06, perturb_amnt=None)
            s = strained.structure
            print('\n\n--->', stype)
            print('\tstrain = ', strained.strain_tensor)
            print('\tpert = ', strained.atomic_perturbations)
        else:
            s = sclass.get_struct_from_structure_type(stype)

        Poscar(s).write_file('POSCAR'+str(stype))

    return










if __name__ == "__main__":
    # print_all_types(straining = True)

    """Testing strained struct generation from structures"""
    sclass = PerfectPerovskite()
    # from pymatgen.analysis.structure_matcher import StructureMatcher
    # sm = StructureMatcher(primitive_cell=False, scale=False, attempt_supercell=False, allow_subset=False)
    for stype in allowed_struct_type:
        print("Trying perturbed structure re-gen for {}".format( stype))
        strain_class = StrainedPerovskite.generate_random_strain(sclass, structure_type=stype,
                                                                 max_strain=0.06, perturb_amnt=None)
        og_strain_struct = strain_class.structure.copy()
        new_strain_class = StrainedPerovskite.generate_from_final_structures( strain_class.base, og_strain_struct)

        # if not sm.fit( og_strain_struct, new_strain_class.structure):
        coord_compare = np.linalg.norm( np.subtract( og_strain_struct.cart_coords.flatten(),
                                        new_strain_class.structure.cart_coords.flatten()))
        og_lattset = list(og_strain_struct.lattice.abc)
        og_lattset.extend( list(og_strain_struct.lattice.angles))
        new_lattset = list(new_strain_class.structure.lattice.abc)
        new_lattset.extend( list(new_strain_class.structure.lattice.angles))
        latt_compare = np.linalg.norm( np.subtract( og_lattset, new_lattset))
        if coord_compare > 1e-5 or latt_compare > 1e-5:
            raise ValueError("Structure fitting failed for {}\ncoord score = {},"
                             " latt score = {}".format( stype, coord_compare, latt_compare))
        else:
            print("\tTest passed!")


