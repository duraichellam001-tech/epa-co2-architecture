

# =========================
# 1. Imports
# =========================

import numpy as np

# =========================
# 2. Constants
# =========================

LB_TO_KG = 0.45359237
METER_PER_MILE = 1609.34
G = 9.81
RHO_AIR = 1.225

LHV_GASOLINE = 44e6          # J/kg
CO2_PER_KG_FUEL = 3.09       # kg CO2 / kg fuel

# =========================
# 3. Drivetrain efficiency
# =========================

def compute_powertrain_efficiency(trans_type, drive_layout):
    eta_engine = 0.35

    eta_trans = {
        "MT": 0.96,
        "AT": 0.92,
        "CVT": 0.94
    }.get(trans_type, 0.92)

    eta_drive = {
        "FWD": 0.98,
        "RWD": 0.97,
        "AWD": 0.95
    }.get(drive_layout, 0.97)

    return eta_engine * eta_trans * eta_drive

# =========================
# 4. Drive cycle loaders
# =========================

def load_ftp_cycle():
    """
    Returns time (s), velocity (m/s)
    """
    # placeholder – we’ll paste validated code
    pass

def load_hwy_cycle():
    """
    Returns time (s), velocity (m/s)
    """
    pass

# =========================
# 5. Vehicle forces
# =========================

def compute_tractive_force(m_kg, v, a, CdA, Crr):
    F_inertia = m_kg * a
    F_roll = m_kg * G * Crr
    F_aero = 0.5 * RHO_AIR * CdA * v**2
    return F_inertia + F_roll + F_aero

# =========================
# 6. Energy & fuel
# =========================

def compute_cycle_fuel_energy(time, velocity, mass_lb, CdA, Crr, eta_pt):
    """
    Returns fuel energy (J)
    """
    # integrate P = F*v
    pass

# =========================
# 7. CO2 conversion
# =========================

def fuel_energy_to_co2_g_per_mi(E_fuel_J, distance_m):
    fuel_kg = E_fuel_J / LHV_GASOLINE
    co2_kg = fuel_kg * CO2_PER_KG_FUEL
    return (co2_kg * 1e3) / (distance_m / METER_PER_MILE)

# =========================
# 8. Public API (THIS is key)
# =========================

def compute_city_hwy_co2_g_per_mi(mass_lb, trans_type, drive_layout):
    """
    Returns:
    {
        "city": value,
        "hwy": value
    }
    """
    pass

def compute_combined_co2_g_per_mi(city, hwy):
    return 0.55 * city + 0.45 * hwy
