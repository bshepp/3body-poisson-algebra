#!/usr/bin/env python3
"""
Scenario definitions for the Multi-System Universality Survey.

Each entry in SCENARIOS defines a physical three-body system with its
potential type, masses, charges, and any extra parameters needed by the
NBodyAlgebra engine.

Categories
----------
gravitational : 1/r potential with various mass hierarchies
atomic        : 1/r Coulomb potential with charges and mass ratios
nuclear       : Yukawa potential (screened 1/r)
plasma        : Logarithmic, Yukawa, or 1/r + external trap
pn            : Post-Newtonian composite potentials
exotic        : 1/r^2 monopoles, dark-matter 1/r
"""

from sympy import Integer, Rational

EXPECTED_SEQUENCE = {0: 3, 1: 6, 2: 17, 3: 116}

SCENARIOS = {
    # =====================================================================
    # GRAVITATIONAL -- 1/r potential, varying mass ratios
    # =====================================================================
    "sun_earth_moon": {
        "category": "gravitational",
        "label": "Sun-Earth-Moon",
        "potential": "1/r",
        "masses": {1: Integer(1), 2: Rational(3, 1000000), 3: Rational(37, 1000000000)},
        "charges": None,
        "potential_params": None,
        "external_potential": None,
        "description": "The original three-body problem -- extreme mass hierarchy",
        "run_atlas": True,
    },
    "sun_jupiter_asteroid": {
        "category": "gravitational",
        "label": "Sun-Jupiter-Asteroid",
        "potential": "1/r",
        "masses": {1: Integer(1), 2: Rational(95, 100000), 3: Rational(1, 10000000000)},
        "charges": None,
        "potential_params": None,
        "external_potential": None,
        "description": "Restricted three-body problem (near-zero asteroid mass)",
        "run_atlas": True,
    },
    "three_cluster_stars": {
        "category": "gravitational",
        "label": "Three Cluster Stars",
        "potential": "1/r",
        "masses": {1: Integer(1), 2: Integer(1), 3: Integer(1)},
        "charges": None,
        "potential_params": None,
        "external_potential": None,
        "description": "Equal-mass stars in a globular cluster (control)",
        "run_atlas": False,
    },
    "binary_star_planet": {
        "category": "gravitational",
        "label": "Binary Star + Planet",
        "potential": "1/r",
        "masses": {1: Integer(1), 2: Integer(1), 3: Rational(1, 1000)},
        "charges": None,
        "potential_params": None,
        "external_potential": None,
        "description": "Symmetric binary with a light planetary companion",
        "run_atlas": True,
    },
    "three_galaxies": {
        "category": "gravitational",
        "label": "Three Merging Galaxies",
        "potential": "1/r",
        "masses": {1: Integer(1), 2: Integer(2), 3: Integer(3)},
        "charges": None,
        "potential_params": None,
        "external_potential": None,
        "description": "Mild mass hierarchy for galaxy-scale merger",
        "run_atlas": False,
    },
    "triple_bh_lisa": {
        "category": "gravitational",
        "label": "Triple Black Holes (LISA)",
        "potential": "1/r",
        "masses": {1: Integer(1), 2: Rational(1, 100), 3: Rational(1, 100000)},
        "charges": None,
        "potential_params": None,
        "external_potential": None,
        "description": "Hierarchical triple BH (10^6:10^4:10 solar masses, normalized)",
        "run_atlas": True,
    },
    "binary_bh_ns": {
        "category": "gravitational",
        "label": "Binary BH + Neutron Star",
        "potential": "1/r",
        "masses": {1: Integer(1), 2: Integer(1), 3: Rational(47, 1000)},
        "charges": None,
        "potential_params": None,
        "external_potential": None,
        "description": "Near-equal BH pair + light NS (30:30:1.4 normalized)",
        "run_atlas": False,
    },

    # =====================================================================
    # ATOMIC / COULOMB -- 1/r with charges and realistic mass ratios
    # =====================================================================
    "helium": {
        "category": "atomic",
        "label": "Helium Atom",
        "potential": "1/r",
        "masses": {1: Integer(7294), 2: Integer(1), 3: Integer(1)},
        "charges": {1: 2, 2: -1, 3: -1},
        "potential_params": None,
        "external_potential": None,
        "description": "He nucleus + 2 electrons (control, already computed)",
        "run_atlas": False,
    },
    "lithium_ion": {
        "category": "atomic",
        "label": "Li+ Ion",
        "potential": "1/r",
        "masses": {1: Integer(12789), 2: Integer(1), 3: Integer(1)},
        "charges": {1: 3, 2: -1, 3: -1},
        "potential_params": None,
        "external_potential": None,
        "description": "Lithium ion -- higher nuclear charge than He",
        "run_atlas": True,
    },
    "h_minus_ion": {
        "category": "atomic",
        "label": "H- Ion",
        "potential": "1/r",
        "masses": {1: Integer(1836), 2: Integer(1), 3: Integer(1)},
        "charges": {1: 1, 2: -1, 3: -1},
        "potential_params": None,
        "external_potential": None,
        "description": "Hydrogen anion -- weakly bound, all attractive pairs",
        "run_atlas": True,
    },
    "positronium_neg": {
        "category": "atomic",
        "label": "Positronium Ps-",
        "potential": "1/r",
        "masses": {1: Integer(1), 2: Integer(1), 3: Integer(1)},
        "charges": {1: 1, 2: -1, 3: -1},
        "potential_params": None,
        "external_potential": None,
        "description": "Equal-mass exotic atom (positron + 2 electrons)",
        "run_atlas": True,
    },
    "muonic_helium": {
        "category": "atomic",
        "label": "Muonic Helium",
        "potential": "1/r",
        "masses": {1: Integer(7294), 2: Integer(1), 3: Integer(207)},
        "charges": {1: 2, 2: -1, 3: -1},
        "potential_params": None,
        "external_potential": None,
        "description": "He nucleus + electron + muon (m_mu ~ 207 m_e)",
        "run_atlas": True,
    },
    "h2_plus_ion": {
        "category": "atomic",
        "label": "H2+ Molecular Ion",
        "potential": "1/r",
        "masses": {1: Integer(1836), 2: Integer(1836), 3: Integer(1)},
        "charges": {1: 1, 2: 1, 3: -1},
        "potential_params": None,
        "external_potential": None,
        "description": "Two repulsive protons + one electron (molecular ion)",
        "run_atlas": True,
    },

    # =====================================================================
    # NUCLEAR -- Yukawa potential (screened 1/r)
    # =====================================================================
    "tritium_he3": {
        "category": "nuclear",
        "label": "Tritium / He-3 (3 Nucleons)",
        "potential": "yukawa",
        "masses": {1: Integer(1), 2: Integer(1), 3: Integer(1)},
        "charges": None,
        "potential_params": [("mu", Rational(7, 10))],
        "external_potential": None,
        "description": "Three nucleons with Yukawa nuclear force (mu ~ 0.7 fm^-1)",
        "run_atlas": True,
    },
    "p_n_n_scattering": {
        "category": "nuclear",
        "label": "Proton-Neutron-Neutron",
        "potential": "yukawa",
        "masses": {1: Integer(1), 2: Rational(10014, 10000), 3: Rational(10014, 10000)},
        "charges": None,
        "potential_params": [("mu", Rational(7, 10))],
        "external_potential": None,
        "description": "p-n-n scattering (proton slightly lighter than neutron)",
        "run_atlas": False,
    },

    # =====================================================================
    # PLASMA / CHARGED PARTICLE
    # =====================================================================
    "penning_trap": {
        "category": "plasma",
        "label": "Three Ions in Penning Trap",
        "potential": "1/r",
        "masses": {1: Integer(1), 2: Integer(1), 3: Integer(1)},
        "charges": {1: 1, 2: 1, 3: 1},
        "potential_params": None,
        "external_potential": {"omega": Integer(1)},
        "description": "Three equal ions (all repulsive) in harmonic trap",
        "run_atlas": True,
    },
    "dusty_plasma": {
        "category": "plasma",
        "label": "Three Dust Grains in Dusty Plasma",
        "potential": "yukawa",
        "masses": {1: Integer(1), 2: Integer(1), 3: Integer(1)},
        "charges": {1: 1, 2: 1, 3: 1},
        "potential_params": [("mu", Rational(1, 10))],
        "external_potential": None,
        "description": "Screened Coulomb / Debye-Yukawa (long Debye length, mu ~ 0.1)",
        "run_atlas": True,
    },
    "two_d_vortices": {
        "category": "plasma",
        "label": "Three 2D Vortices",
        "potential": "log",
        "masses": {1: Integer(1), 2: Integer(1), 3: Integer(1)},
        "charges": None,
        "potential_params": None,
        "external_potential": None,
        "description": "Point vortices in 2D fluid with logarithmic interaction",
        "run_atlas": True,
    },

    # =====================================================================
    # POST-NEWTONIAN / GR
    # =====================================================================
    "kozai_lidov": {
        "category": "pn",
        "label": "Kozai-Lidov (1PN Hierarchical)",
        "potential": "composite",
        "masses": {1: Integer(1), 2: Rational(1, 100), 3: Rational(1, 100000)},
        "charges": None,
        "potential_params": [(-Integer(1), 1), (Rational(-1, 200), 2)],
        "external_potential": None,
        "description": "Hierarchical triple with 1PN GR precession (c=10)",
        "run_atlas": False,
    },

    # =====================================================================
    # EXOTIC
    # =====================================================================
    "magnetic_monopoles": {
        "category": "exotic",
        "label": "Three Magnetic Monopoles",
        "potential": "1/r^2",
        "masses": {1: Integer(1), 2: Integer(1), 3: Integer(1)},
        "charges": None,
        "potential_params": None,
        "external_potential": None,
        "description": "Hypothetical magnetic monopoles with 1/r^2 interaction",
        "run_atlas": True,
    },
    "dark_matter": {
        "category": "exotic",
        "label": "Dark Matter Halo Scattering",
        "potential": "1/r",
        "masses": {1: Integer(1), 2: Integer(5), 3: Integer(10)},
        "charges": None,
        "potential_params": None,
        "external_potential": None,
        "description": "Three dark matter halos with gravitational interaction",
        "run_atlas": False,
    },
}


DOCUMENTED_ONLY = {
    "three_quarks": {
        "label": "Three Quarks (QCD)",
        "potential_form": "V ~ -alpha_s/r (short) + sigma*r (long, confining)",
        "reason_skipped": (
            "The confining linear potential V = sigma*r is fundamentally "
            "non-perturbative. Our polynomial u = 1/r representation cannot "
            "naturally express linear-in-r terms. More importantly, the "
            "three-quark system requires color SU(3) gauge theory, not "
            "classical pairwise potentials."
        ),
        "pathway": (
            "A limited exploration could add a composite potential with "
            "(-1, 1) for the Coulombic short-range piece and approximate "
            "the linear confinement via a polynomial fit in 1/u. However, "
            "physical relevance is questionable without quantum/gauge effects. "
            "A more principled approach would use lattice QCD potentials "
            "tabulated numerically."
        ),
    },
    "three_anyons": {
        "label": "Three Anyons in 2D",
        "potential_form": "Fractional statistics -- not a classical potential",
        "reason_skipped": (
            "Anyonic statistics are intrinsically quantum mechanical. In 2D, "
            "exchanging two anyons multiplies the wavefunction by e^{i*theta} "
            "where theta is neither 0 (bosons) nor pi (fermions). The "
            "classical Poisson algebra has no notion of particle exchange "
            "statistics; {f, g} is the same regardless of identical-particle "
            "symmetry."
        ),
        "pathway": (
            "Deformation quantization replaces the Poisson bracket {f, g} "
            "with the Moyal star-commutator [f, g]_star = f*g - g*f. "
            "Anyonic statistics could then be imposed by restricting the "
            "Hilbert space to the appropriate representation of the braid "
            "group. This would be a fundamental extension of the algebraic "
            "framework from classical to quantum."
        ),
    },
}


CATEGORIES = [
    ("gravitational", "Gravitational (1/r)"),
    ("atomic", "Atomic / Coulomb (1/r)"),
    ("nuclear", "Nuclear (Yukawa)"),
    ("plasma", "Plasma / Charged Particle"),
    ("pn", "Post-Newtonian / GR"),
    ("exotic", "Exotic"),
]


def get_scenarios_by_category(category):
    """Return scenario keys for a given category."""
    return [k for k, v in SCENARIOS.items() if v["category"] == category]


def get_atlas_scenarios():
    """Return scenario keys that should have atlas scans."""
    return [k for k, v in SCENARIOS.items() if v.get("run_atlas", False)]
