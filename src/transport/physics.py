"""Evidence-based transport closure for an intact TH21 particle.

Mass is expressed as ng of 252-nt TH21 core DNA per cm2. The two fitted
coefficients describe the follicular lumen and lateral wall. Diffusion after
wall exit is anchored to measured human-skin dextran transport.
"""
from dataclasses import dataclass
import numpy as np
from scipy.special import i0

KB=1.380649e-23
NA=6.02214076e23
ECH=1.602176634e-19
EPS0=8.8541878128e-12
CORE_MW=252*330.0
MASS_NG=CORE_MW/NA*1e9

@dataclass(frozen=True)
class Inputs:
    application_min:float=15.0
    target_uM:float=0.5
    required_contact_min:float=15.0
    loading_fraction:float=1.0
    diameter_nm:float=17.02
    zeta_mV:float=-8.188
    temperature_K:float=305.15
    viscosity_Pas:float=0.000765
    relative_permittivity:float=76.0
    ionic_strength_M:float=0.15
    follicles_cm2:float=292.0
    volume_single_cm3:float=0.00019/292
    wall_single_cm2:float=0.137/292
    orifice_diameter_um:float=66.0
    follicle_length_um:float=225.0
    follicle_viscosity_ratio:float=691.8079913260484
    wall_P_ref_cm_h:float=2.289443694622724e-5
    packing_fraction:float=0.64
    dextran40_radius_nm:float=4.5
    dextran500_radius_nm:float=13.5
    dextran40_ve_um2_s:float=11.5
    dextran500_ve_um2_s:float=8.0
    dextran40_dermis_um2_s:float=23.0
    dextran500_dermis_um2_s:float=9.0
    collagen_zeta_mV:float=18.37
    epsilon_ve:float=0.12
    epsilon_dermis:float=0.40
    internalization_hazard_h:float=0.07700660339137827
    intact_half_h:float=26.0
    sc_um:float=20.0
    ve_bottom_um:float=100.0
    bottom_um:float=400.0
    dx_um:float=2.5
    follicle_dx_um:float=6.25

def stokes(p,diameter_nm=None,viscosity_ratio=1.0):
    d=p.diameter_nm if diameter_nm is None else diameter_nm
    return KB*p.temperature_K/(3*np.pi*p.viscosity_Pas*viscosity_ratio*d*1e-9)*1e4*3600

def debye_nm(p):
    return np.sqrt(EPS0*p.relative_permittivity*KB*p.temperature_K/(2*NA*1000*p.ionic_strength_M*ECH**2))*1e9

def collagen_energy_kBT(p):
    """Screened sphere-plane energy at one measured-salt screening length."""
    a=p.diameter_nm*0.5e-9
    return 4*np.pi*EPS0*p.relative_permittivity*a*(p.zeta_mV*1e-3)*(p.collagen_zeta_mV*1e-3)*np.exp(-1)/(KB*p.temperature_K)

def charge_mobility(p):
    """Lifson-Jackson periodic-potential correction, normalized at TH21 charge.

    The energy oscillates between zero and the screened contact energy. A
    constant energy offset cancels, leaving I0(U/2)^-2. This is the explicit
    charge-dependence of the extracellular diffusion comparison.
    """
    u=collagen_energy_kBT(p)
    return 1/i0(u/2)**2

def skin_diffusion(p,layer):
    a=p.diameter_nm/2
    r1,r2=p.dextran40_radius_nm,p.dextran500_radius_nm
    if layer=='VE':d1,d2=p.dextran40_ve_um2_s,p.dextran500_ve_um2_s
    else:d1,d2=p.dextran40_dermis_um2_s,p.dextran500_dermis_um2_s
    exponent=np.log(d2/d1)/np.log(r2/r1)
    d=d1*(a/r1)**exponent
    ref_charge=charge_mobility(Inputs())
    d*=charge_mobility(p)/ref_charge
    return dict(D_cm2_h=float(d*3.6e-5),D_um2_s=float(d),size_exponent=float(exponent),partition=1.0,charge_factor=float(charge_mobility(p)/ref_charge))

def derived(p):
    ve=skin_diffusion(p,'VE');derm=skin_diffusion(p,'dermis')
    a_cm=p.diameter_nm*.5e-7
    packing_density=p.packing_fraction*MASS_NG/(4*np.pi*a_cm**3/3)
    V=p.follicles_cm2*p.volume_single_cm3
    A=p.follicles_cm2*p.wall_single_cm2
    open_area=p.follicles_cm2*np.pi*(p.orifice_diameter_um*.5e-4)**2
    # Classical access resistance for diffusion into a circular opening.
    entry_length_um=np.pi*p.orifice_diameter_um/8
    # The fitted wall permeability is anchored at the reference TH21 size and
    # charge; other descriptors scale by Brownian mobility and screened charge.
    ref=Inputs()
    scale=stokes(p)/stokes(ref)*charge_mobility(p)/charge_mobility(ref)
    return dict(D0_cm2_h=stokes(p),Debye_nm=debye_nm(p),VE=ve,dermis=derm,
        follicle_D_cm2_h=stokes(p,viscosity_ratio=p.follicle_viscosity_ratio),
        follicle_volume_cm3_cm2=V,wall_area_cm2_cm2=A,
        follicle_capacity_ng_cm2=V*packing_density,packing_density_ng_ml=packing_density,
        entry_area_fraction=open_area,entry_length_um=entry_length_um,
        wall_P_cm_h=p.wall_P_ref_cm_h*scale,wall_P_scale=scale,
        uptake_h=p.internalization_hazard_h,
        molecular_weight_g_mol=CORE_MW,mass_particle_ng=MASS_NG,
        collagen_energy_kBT=collagen_energy_kBT(p))
