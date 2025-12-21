"""
physics_co2.py

Deterministic physics-based CO₂ estimation
for early-phase vehicle architecture screening.

- Importable by ML training & inference
- Executable as a standalone sanity-check script
"""

# =========================
# Imports
# =========================

import numpy as np
import os

# =========================
# Constants
# =========================

LB_TO_KG = 0.45359237
METER_PER_MILE = 1609.34

G = 9.81
RHO_AIR = 1.225

LHV_GASOLINE = 44e6
CO2_PER_KG_FUEL = 3.09

# =========================
# Drivetrain efficiency
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
# Drive cycle loading
# =========================

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
CYCLE_DIR = os.path.join(_THIS_DIR, "cycles")

def load_ftp_cycle():
    data = np.loadtxt(
        os.path.join(CYCLE_DIR, "EPA_CITY.csv"),
        delimiter=",",
        skiprows=1
    )
    return data[:, 0], data[:, 1]

def load_hwy_cycle():
    data = np.loadtxt(
        os.path.join(CYCLE_DIR, "EPA_HWY.csv"),
        delimiter=",",
        skiprows=1
    )
    return data[:, 0], data[:, 1]

# =========================
# Physics core
# =========================
def compute_CdA(mass_lb, CdA_ref=0.65, m_ref_lb=4000):
    m_ratio = mass_lb / m_ref_lb
    return CdA_ref * m_ratio ** (2/3)


def compute_Crr(mass_lb, Crr_ref=0.010, m_ref_lb=4000):
    m_ratio = mass_lb / m_ref_lb
    return Crr_ref * (1 + 0.05 * np.log(m_ratio))

def compute_acceleration(t, v):
    return np.gradient(v, t)

def compute_distance(t, v):
    return np.trapz(v, t)

def compute_tractive_force(m_kg, v, a, CdA, Crr):
    F_inertia = m_kg * a
    F_roll = m_kg * G * Crr
    F_aero = 0.5 * RHO_AIR * CdA * v**2
    return F_inertia + F_roll + F_aero

def compute_tractive_energy(t, v, F):
    P = F * v
    P[P < 0] = 0.0  # no regen modeled
    return np.trapz(P, t)

def compute_city_hwy_co2_g_per_mi(mass_lb, trans_type, drive_layout):
    """
    Physics-based CO₂ estimation for EPA FTP & HWY cycles.

    Returns
    -------
    dict with keys ['city', 'hwy']
    """
    # --- conversions ---
    m_kg = mass_lb * LB_TO_KG

    # --- mass-anchored physics ---
    CdA = compute_CdA(mass_lb)
    Crr = compute_Crr(mass_lb)

    # --- powertrain ---
    eta_pt = compute_powertrain_efficiency(trans_type, drive_layout)

    results = {}

    cycles = {
        "city": load_ftp_cycle,
        "hwy": load_hwy_cycle,
    }

    for label, loader in cycles.items():
        t, v = loader()

        a = compute_acceleration(t, v)
        F = compute_tractive_force(m_kg, v, a, CdA, Crr)
        E_trac = compute_tractive_energy(t, v, F)

        # fuel energy (ICE, no regen)
        k_cycle = 0.60 if label == "EPA_CITY" else 0.70
        E_fuel = (E_trac / eta_pt) * k_cycle


        # distance
        dist_m = compute_distance(t, v)

        # CO₂
        fuel_kg = E_fuel / LHV_GASOLINE
        co2_kg = fuel_kg * CO2_PER_KG_FUEL

        co2_gpm = (co2_kg * 1e3) / (dist_m / METER_PER_MILE)
        results[label] = co2_gpm

    return results


def compute_combined_co2_g_per_mi(city, hwy):
    return 0.55 * city + 0.45 * hwy

# =========================
# Single-shot execution
# =========================

def main():
    """
    Sanity-check run for physics baseline.
    """
    print("Running physics_co2 sanity check...\n")

    mass_lb = 4000
    trans = "AT"
    drive = "FWD"

    res = compute_city_hwy_co2_g_per_mi(
        mass_lb=mass_lb,
        trans_type=trans,
        drive_layout=drive
    )

    city = res["city"]
    hwy = res["hwy"]
    comb = compute_combined_co2_g_per_mi(city, hwy)

    print(f"Mass           : {mass_lb} lb")
    print(f"Transmission   : {trans}")
    print(f"Drive layout   : {drive}\n")

    print(f"City CO₂ (g/mi): {city:.1f}")
    print(f"Hwy  CO₂ (g/mi): {hwy:.1f}")
    print(f"Comb CO₂ (g/mi): {comb:.1f}")

    print("\nPhysics sanity check DONE.")


if __name__ == "__main__":
    main()
