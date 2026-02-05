import streamlit as st
import pandas as pd
import numpy as np
import math
import sys
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
from datetime import datetime

# PDF Generation with ReportLab (more powerful than fpdf)
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor, white
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, 
    PageBreak, KeepTogether
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="CAT Size Solution", page_icon="⚡", layout="wide")

# ==============================================================================
# 0. HYBRID DATA LIBRARY - ENHANCED WITH DENSITY & RAMP RATE
# ==============================================================================

leps_gas_library = {
    "XGC1900": {
        "description": "Mobile Power Module (High Speed)",
        "type": "High Speed",
        "iso_rating_mw": 1.9,
        "electrical_efficiency": 0.392,
        "heat_rate_lhv": 8780,
        "step_load_pct": 25.0, 
        "ramp_rate_mw_s": 0.5,  # NEW
        "emissions_nox": 0.5,
        "emissions_co": 2.5,
        "mtbf_hours": 50000,  # NEW: Mean Time Between Failures
        "maintenance_interval_hrs": 1000,  # Time between planned maintenance
        "maintenance_duration_hrs": 48,    # Downtime per maintenance event
        "default_for": 2.0, 
        "default_maint": 5.0,
        "est_cost_kw": 775.0,      
        "est_install_kw": 300.0,
        "power_density_mw_per_m2": 0.010,  # NEW: 200 m²/MW
        "gas_pressure_min_psi": 1.5,
        "reactance_xd_2": 0.14
    },
    "G3520FR": {
        "description": "Fast Response Gen Set (High Speed)",
        "type": "High Speed",
        "iso_rating_mw": 2.5,
        "electrical_efficiency": 0.386,
        "heat_rate_lhv": 8836,
        "step_load_pct": 40.0,
        "ramp_rate_mw_s": 0.6,
        "emissions_nox": 0.5,
        "emissions_co": 2.1,
        "mtbf_hours": 48000,
        "maintenance_interval_hrs": 1000,
        "maintenance_duration_hrs": 48,
        "default_for": 2.0,
        "default_maint": 5.0,
        "est_cost_kw": 575.0,
        "est_install_kw": 650.0,
        "power_density_mw_per_m2": 0.010,
        "gas_pressure_min_psi": 1.5,
        "reactance_xd_2": 0.14
    },
    "G3520K": {
        "description": "High Efficiency Gen Set (High Speed)",
        "type": "High Speed",
        "iso_rating_mw": 2.4,
        "electrical_efficiency": 0.453,
        "heat_rate_lhv": 7638,
        "step_load_pct": 15.0,
        "ramp_rate_mw_s": 0.4,
        "emissions_nox": 0.3,
        "emissions_co": 2.3,
        "mtbf_hours": 52000,
        "maintenance_interval_hrs": 1000,
        "maintenance_duration_hrs": 48,
        "default_for": 2.5,
        "default_maint": 6.0,
        "est_cost_kw": 575.0,
        "est_install_kw": 650.0,
        "power_density_mw_per_m2": 0.010,
        "gas_pressure_min_psi": 1.5,
        "reactance_xd_2": 0.13
    },
    "CG260-16": {
        "description": "Cogeneration Specialist (High Speed)",
        "type": "High Speed",
        "iso_rating_mw": 3.96,
        "electrical_efficiency": 0.434,
        "heat_rate_lhv": 7860,
        "step_load_pct": 10.0,
        "ramp_rate_mw_s": 0.45,
        "emissions_nox": 0.5,
        "emissions_co": 1.8,
        "mtbf_hours": 55000,
        "maintenance_interval_hrs": 1000,
        "maintenance_duration_hrs": 48,
        "default_for": 3.0,
        "default_maint": 5.0,
        "est_cost_kw": 675.0,
        "est_install_kw": 1100.0,
        "power_density_mw_per_m2": 0.009,
        "gas_pressure_min_psi": 7.25,
        "reactance_xd_2": 0.15
    },
    "Titan 130": {
        "description": "Solar Gas Turbine (16.5 MW)",
        "type": "Gas Turbine",
        "iso_rating_mw": 16.5,
        "electrical_efficiency": 0.354,
        "heat_rate_lhv": 9630,
        "step_load_pct": 15.0,
        "ramp_rate_mw_s": 2.0,  # Turbines ramp faster
        "emissions_nox": 0.6,
        "emissions_co": 0.6,
        "mtbf_hours": 80000,  # Turbines more reliable
        "maintenance_interval_hrs": 8000,  # Less frequent maintenance
        "maintenance_duration_hrs": 120,   # Longer downtime per event
        "default_for": 1.5,
        "default_maint": 2.0,
        "est_cost_kw": 775.0,
        "est_install_kw": 1000.0,
        "power_density_mw_per_m2": 0.020,  # NEW: 50% less footprint
        "gas_pressure_min_psi": 300.0,
        "reactance_xd_2": 0.18
    },
    "G20CM34": {
        "description": "Medium Speed Baseload Platform",
        "type": "Medium Speed",
        "iso_rating_mw": 9.76,
        "electrical_efficiency": 0.475,
        "heat_rate_lhv": 7480,
        "step_load_pct": 10.0,
        "ramp_rate_mw_s": 0.3,  # Slower ramp
        "emissions_nox": 0.5,
        "emissions_co": 0.5,
        "mtbf_hours": 60000,
        "maintenance_interval_hrs": 2500,  # Medium speed: less frequent
        "maintenance_duration_hrs": 72,    # Longer downtime
        "default_for": 3.0, 
        "default_maint": 5.0,
        "est_cost_kw": 700.0,
        "est_install_kw": 1250.0,
        "power_density_mw_per_m2": 0.008,  # Larger footprint
        "gas_pressure_min_psi": 90.0,
        "reactance_xd_2": 0.16
    }
}

# ==============================================================================
# HELPER FUNCTIONS - ENHANCED
# ==============================================================================

def get_part_load_efficiency(base_eff, load_pct, gen_type):
    """
    Efficiency curves validated against CAT test data using Linear Interpolation.
    Ensures 100% load = 100% of rated efficiency.
    """
    # Clamp load percentage reasonable limits
    load_pct = max(0, min(100, load_pct))
    
    if gen_type == "High Speed":
        # Data points (Load %, Efficiency Factor)
        # Based on G3520/G3516 characteristic curves
        xp = [0, 25, 50, 75, 100]
        fp = [0.0, 0.70, 0.88, 0.96, 1.00]
        
    elif gen_type == "Medium Speed":
        # Medium speed engines (G20CM34) are flatter at part load
        xp = [0, 25, 50, 75, 100]
        fp = [0.0, 0.75, 0.91, 0.97, 1.00]
        
    elif gen_type == "Gas Turbine":
        # Turbines lose efficiency very fast at part load
        xp = [0, 25, 50, 75, 100]
        fp = [0.0, 0.55, 0.78, 0.90, 1.00]
        
    else:
        return base_eff

    # Interpolate exact factor
    factor = np.interp(load_pct, xp, fp)
    
    return base_eff * factor

def transient_stability_check(xd_pu, num_units, step_load_pct):
    """Critical voltage sag check for AI workloads"""
    equiv_xd = xd_pu / math.sqrt(num_units)
    voltage_sag = (step_load_pct/100) * equiv_xd * 100
    if voltage_sag > 10:
        return False, voltage_sag
    return True, voltage_sag

# ==============================================================================
# NEW: SPINNING RESERVE CALCULATION FUNCTION
# ==============================================================================

def calculate_spinning_reserve_units(p_avg_load, unit_capacity, spinning_reserve_pct, 
                                     use_bess=False, bess_power_mw=0, gen_step_capability_pct=0):
    """
    CORRECTED: Calculate the number of running units considering spinning reserve.
    
    SPINNING RESERVE = Extra online capacity to handle sudden load increases or generator trips.
    This capacity must be ONLINE and ready to respond instantly.
    
    WITHOUT BESS:
    - Generators must provide ALL spinning reserve as HEADROOM
    - Online Capacity = Avg Load + Full Spinning Reserve
    - More units running at LOWER load per unit
    
    WITH BESS:
    - BESS provides instant response for spinning reserve
    - Online Capacity = Avg Load + small margin (BESS covers the reserve)
    - Fewer units running at HIGHER load per unit = better efficiency
    
    Example (110 MW avg, 40% spinning reserve = 44 MW, 1.9 MW generators):
    
    WITHOUT BESS:
      - Required online capacity = 110 + 44 = 154 MW
      - Units needed = ceil(154 / 1.9) = 82 units
      - Load/Unit = 110 / (82 * 1.9) = 70.6%
    
    WITH BESS (44 MW):
      - BESS covers spinning reserve
      - Required online capacity = 110 * 1.05 = 115.5 MW (5% margin)
      - Units needed = ceil(115.5 / 1.9) = 61 units
      - Load/Unit = 110 / (61 * 1.9) = 94.9%
    """
    
    # Step 1: Calculate spinning reserve requirement in MW
    spinning_reserve_mw = p_avg_load * (spinning_reserve_pct / 100)
    
    # Step 2: Determine how BESS contributes to spinning reserve
    if use_bess and bess_power_mw > 0:
        # BESS can cover spinning reserve up to its power rating
        spinning_from_bess = min(bess_power_mw, spinning_reserve_mw)
    else:
        spinning_from_bess = 0
    
    # Step 3: Remaining spinning reserve that generators must provide as HEADROOM
    spinning_from_gens = spinning_reserve_mw - spinning_from_bess
    
    # Step 4: Calculate required online capacity
    # This is the KEY difference between BESS and No-BESS scenarios
    if use_bess and spinning_from_bess >= spinning_reserve_mw * 0.9:
        # BESS covers most/all spinning reserve
        # Generators only need to cover average load + small margin
        required_online_capacity = p_avg_load * 1.05  # 5% operational margin
    else:
        # Generators must have HEADROOM for spinning reserve
        # Online capacity = Average Load + Generator portion of spinning reserve
        required_online_capacity = p_avg_load + spinning_from_gens
    
    # Step 5: Calculate number of running units
    # Simple: enough units to provide required online capacity
    n_units_running = math.ceil(required_online_capacity / unit_capacity)
    
    # Ensure at least 1 unit
    n_units_running = max(1, n_units_running)
    
    # Step 6: Calculate actual metrics
    total_online_capacity = n_units_running * unit_capacity
    load_per_unit_pct = (p_avg_load / total_online_capacity) * 100
    headroom_available = total_online_capacity - p_avg_load
    
    return {
        'n_units_running': n_units_running,
        'load_per_unit_pct': load_per_unit_pct,
        'spinning_reserve_mw': spinning_reserve_mw,
        'spinning_from_gens': spinning_from_gens,
        'spinning_from_bess': spinning_from_bess,
        'total_online_capacity': total_online_capacity,
        'headroom_available': headroom_available,
        'required_online_capacity': required_online_capacity
    }

def calculate_bess_requirements(p_net_req_avg, p_net_req_peak, step_load_req, 
                                gen_ramp_rate, gen_step_capability, 
                                load_change_rate_req,  # <--- NUEVO ARGUMENTO
                                enable_black_start=False):
    """
    Sophisticated BESS sizing based on actual transient analysis
    """
    # Component 1: Step Load Support
    step_load_mw = p_net_req_avg * (step_load_req / 100)
    gen_step_mw = p_net_req_avg * (gen_step_capability / 100)
    bess_step_support = max(0, step_load_mw - gen_step_mw)
    
    # Component 2: Peak Shaving
    bess_peak_shaving = p_net_req_peak - p_net_req_avg
    
    # Component 3: Ramp Rate Support (AHORA DINÁMICO)
    # Antes: load_change_rate = 5.0  <--- ELIMINADO
    # Ahora usamos la variable del input:
    bess_ramp_support = max(0, (load_change_rate_req - gen_ramp_rate) * 10)  # 10s buffer
    
    # ... (Resto de componentes igual: Freq Reg, Black Start, Spinning) ...
    bess_freq_reg = p_net_req_avg * 0.05
    bess_black_start = p_net_req_peak * 0.05 if enable_black_start else 0
    bess_spinning_reserve = p_net_req_avg * (step_load_req / 100)
    
    # Total Power
    bess_power_total = max(
        bess_step_support,
        bess_peak_shaving,
        bess_ramp_support,
        bess_freq_reg,
        bess_black_start,
        bess_spinning_reserve,
        p_net_req_peak * 0.15
    )
    
    # ... (Cálculo de energía y retorno igual) ...
    c_rate = 1.0
    bess_energy_total = bess_power_total / c_rate / 0.85
    
    breakdown = {
        'step_support': bess_step_support,
        'peak_shaving': bess_peak_shaving,
        'ramp_support': bess_ramp_support,
        'freq_reg': bess_freq_reg,
        'black_start': bess_black_start,
        'spinning_reserve': bess_spinning_reserve
    }
    
    return bess_power_total, bess_energy_total, breakdown

def calculate_bess_reliability_credit(bess_power_mw, bess_energy_mwh, 
                                      unit_capacity_mw, mttr_hours=48):
    """
    Calculate how many genset equivalents BESS can replace for reliability
    
    IMPORTANT: BESS cannot cover full MTTR (48 hrs), only provides bridge power
    while backup gensets start up (typically 2-4 hours)
    
    Parameters:
    - bess_power_mw: BESS power capacity (MW)
    - bess_energy_mwh: BESS energy capacity (MWh)
    - unit_capacity_mw: Single generator capacity (MW)
    - mttr_hours: Mean Time To Repair (default 48 hrs, but BESS covers 2-4 hrs)
    
    Returns:
    - effective_credit: Number of gensets BESS can replace (float)
    - credit_breakdown: Dictionary with calculation details
    """
    if bess_power_mw <= 0 or bess_energy_mwh <= 0:
        return 0.0, {}
    
    # BESS realistically covers 2-4 hours until backup gensets arrive or repair starts
    # NOT the full MTTR of 48 hours
    realistic_coverage_hrs = 2.0  # Conservative: 2 hours
    
    # Power-based credit: How many units can BESS replace instantly
    power_credit = bess_power_mw / unit_capacity_mw
    
    # Energy-based credit: How long can BESS sustain that power
    bess_duration_hrs = bess_energy_mwh / bess_power_mw if bess_power_mw > 0 else 0
    energy_credit = bess_energy_mwh / (unit_capacity_mw * realistic_coverage_hrs)
    
    # Take minimum (bottleneck)
    raw_credit = min(power_credit, energy_credit)
    
    # Apply factors:
    bess_availability = 0.98  # BESS itself has ~98% availability
    coverage_factor = 0.70    # Increased from 0.60 to 0.70 (less conservative)
    
    effective_credit = raw_credit * bess_availability * coverage_factor
    
    credit_breakdown = {
        'power_credit': power_credit,
        'energy_credit': energy_credit,
        'raw_credit': raw_credit,
        'bess_availability': bess_availability,
        'coverage_factor': coverage_factor,
        'effective_credit': effective_credit,
        'bess_duration_hrs': bess_duration_hrs,
        'realistic_coverage_hrs': realistic_coverage_hrs
    }
    
    return effective_credit, credit_breakdown

def calculate_availability_weibull(n_total, n_running, mtbf_hours, project_years, 
                                  maintenance_interval_hrs=1000, maintenance_duration_hrs=48):
    """
    Reliability model using industry standard availability formula INCLUDING planned maintenance
    
    Availability = MTBF / (MTBF + MTTR + Planned_Maintenance_Time)
    
    This is the CORRECT formula that accounts for:
    1. Random failures (MTBF/MTTR)
    2. Planned maintenance outages (often overlooked!)
    
    For High Speed RICE:
    - Maintenance every 1000 hrs → 8.76 events/year
    - 48 hrs per event → 420 hrs/year unavailable
    - This is 4.8% unavailability just from maintenance!
    """
    # Typical MTTR for power generation equipment
    mttr_hours = 48  # 48 hours average repair time for failures
    
    # Calculate planned maintenance unavailability
    # Annual hours of planned maintenance
    annual_maintenance_hrs = (8760 / maintenance_interval_hrs) * maintenance_duration_hrs
    
    # Total unavailability = MTTR (failures) + Planned Maintenance
    total_unavailable_hrs = mttr_hours + annual_maintenance_hrs
    
    # Unit availability (corrected formula)
    unit_availability = mtbf_hours / (mtbf_hours + total_unavailable_hrs)
    
    # Example for High Speed RICE (MTBF=48000, maint every 1000 hrs for 48 hrs):
    # annual_maintenance = (8760/1000) × 48 = 420.5 hrs
    # availability = 48000 / (48000 + 48 + 420.5) = 48000 / 48468.5 = 99.03%
    # vs OLD (incorrect): 48000 / 48048 = 99.90% ← 0.87% too optimistic!
    
    # System availability (N+X configuration using binomial)
    # P(system works) = P(at least n_running units are available)
    sys_avail = 0
    for k in range(n_running, n_total + 1):
        comb = math.comb(n_total, k)
        prob = comb * (unit_availability ** k) * ((1 - unit_availability) ** (n_total - k))
        sys_avail += prob
    
    # For availability curve over time, apply modest aging (0.1% per year)
    availability_over_time = []
    for year in range(1, project_years + 1):
        # Conservative aging: 0.1% per year
        aging_factor = 1.0 - (year * 0.001)
        aging_factor = max(0.95, aging_factor)  # Floor at 95%
        
        aged_unit_availability = unit_availability * aging_factor
        
        # Recalculate system availability with aged units
        sys_avail_year = 0
        for k in range(n_running, n_total + 1):
            comb = math.comb(n_total, k)
            prob = comb * (aged_unit_availability ** k) * ((1 - aged_unit_availability) ** (n_total - k))
            sys_avail_year += prob
        
        availability_over_time.append(sys_avail_year)
    
    # Return year 1 availability (not average over 20 years)
    # This is standard practice for availability targets
    return sys_avail, availability_over_time

def optimize_fleet_size(p_net_req_avg, p_net_req_peak, unit_cap, step_load_req, gen_data, use_bess=False):
    """
    Multi-objective fleet optimization
    NOW CONSIDERS: BESS for peak shaving and step load coverage
    """
    # NEW: If BESS enabled, size for average load (BESS handles peaks)
    if use_bess:
        # Constraint 1: Average capacity + safety margin (BESS covers peak)
        n_min_peak = math.ceil(p_net_req_avg * 1.15 / unit_cap)  # +15% margin
        
        # Constraint 3: Step load covered by BESS (not gensets)
        # Only need headroom for ramp-up capability
        headroom_required = p_net_req_avg * 1.10  # 10% headroom only
        n_min_step = math.ceil(headroom_required / unit_cap)
    else:
        # Constraint 1: Must cover peak load without BESS
        n_min_peak = math.ceil(p_net_req_peak / unit_cap)
        
        # Constraint 3: Must have headroom for step load
        headroom_required = p_net_req_avg * (1 + step_load_req/100) * 1.20
        n_min_step = math.ceil(headroom_required / unit_cap)
    
    # Constraint 2: Part-load efficiency (target 65-80% load)
    n_ideal_eff = math.ceil(p_net_req_avg / (unit_cap * 0.72))
    
    # Take maximum of all constraints
    n_running_optimal = max(n_min_peak, n_ideal_eff, n_min_step)
    
    # Analyze efficiency at different fleet sizes
    fleet_options = {}
    for n in range(max(1, n_running_optimal - 1), n_running_optimal + 3):
        # Check if meets minimum capacity
        if use_bess:
            if n * unit_cap < p_net_req_avg * 1.10:  # Need 110% of average
                continue
        else:
            if n * unit_cap < p_net_req_peak:  # Need full peak
                continue
        
        load_pct = (p_net_req_avg / (n * unit_cap)) * 100
        if load_pct < 30 or load_pct > 95:  # Outside acceptable range
            continue
        eff = get_part_load_efficiency(gen_data["electrical_efficiency"], load_pct, gen_data["type"])
        
        # Score: favor 65-80% load range
        optimal_load = 72.5
        load_penalty = abs(load_pct - optimal_load) / 100
        fleet_options[n] = {
            'efficiency': eff,
            'load_pct': load_pct,
            'score': eff * (1 - load_penalty * 0.5)
        }
    
    if fleet_options:
        optimal_n = max(fleet_options, key=lambda x: fleet_options[x]['score'])
        return optimal_n, fleet_options
    else:
        return n_running_optimal, {}
    
    # Take maximum
    n_running_optimal = max(n_min_peak, n_ideal_eff, n_min_step)
    
    # Analyze efficiency at different fleet sizes
    fleet_options = {}
    for n in range(max(1, n_running_optimal - 1), n_running_optimal + 3):
        if n * unit_cap < p_net_req_peak:
            continue
        load_pct = (p_net_req_avg / (n * unit_cap)) * 100
        if load_pct < 20 or load_pct > 95:  # Outside acceptable range
            continue
        eff = get_part_load_efficiency(gen_data["electrical_efficiency"], load_pct, gen_data["type"])
        fleet_options[n] = {
            'efficiency': eff,
            'load_pct': load_pct,
            'score': eff * (1 - abs(load_pct - 70)/100)  # Penalize deviation from 70%
        }
    
    if fleet_options:
        optimal_n = max(fleet_options, key=lambda x: fleet_options[x]['score'])
        return optimal_n, fleet_options
    else:
        return n_running_optimal, {}

def calculate_macrs_depreciation(capex, project_years):
    """
    MACRS 5-year depreciation schedule
    """
    macrs_schedule = [0.20, 0.32, 0.192, 0.1152, 0.1152, 0.0576]
    tax_rate = 0.21  # Federal corporate tax rate
    
    pv_benefit = 0
    wacc = 0.08  # Use global WACC
    
    for year, rate in enumerate(macrs_schedule, 1):
        if year > project_years:
            break
        annual_benefit = capex * rate * tax_rate
        pv_benefit += annual_benefit / ((1 + wacc) ** year)
    
    return pv_benefit

# ==============================================================================
# 1. GLOBAL SETTINGS & SIDEBAR (ORGANIZED v3.1)
# ==============================================================================

with st.sidebar:
    st.image("https://img.icons8.com/color/96/generator.png", width=60)
    st.header("Global Settings")
    c_glob1, c_glob2 = st.columns(2)
    unit_system = c_glob1.radio("Units", ["Metric (SI)", "Imperial (US)"], label_visibility="collapsed")
    freq_hz = c_glob2.radio("Freq", [60, 50], label_visibility="collapsed")

    is_imperial = "Imperial" in unit_system
    is_50hz = freq_hz == 50
    
    # Definición de Unidades
    if is_imperial:
        u_temp, u_dist, u_area = "°F", "ft", "ft²"
        u_press = "psig"
    else:
        u_temp, u_dist, u_area = "°C", "m", "m²"
        u_press = "Bar"

    st.title(f"CAT Size Solution ({freq_hz}Hz)")
    st.divider()

    # -------------------------------------------------------------------------
    # 1. PERFIL DE CARGA (LOAD PROFILE)
    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    # 1. PERFIL DE CARGA (LOAD PROFILE)
    # -------------------------------------------------------------------------
    st.header("1. Load Profile")
    
    dc_type = st.selectbox("Data Center Type", [
        "AI Factory (Training)", "AI Factory (Inference)", 
        "Hyperscale Standard", "Colocation", "Edge Computing"
    ])
    
    p_it = st.number_input("Critical IT Load (MW)", 1.0, 2000.0, 100.0, step=10.0)
    
    # --- VARIABLE FALTANTE AGREGADA AQUÍ ---
    avail_req = st.number_input("Target Availability (%)", 90.0, 99.9999, 99.99, format="%.4f", help="Reliability Target (Defines N+X redundancy)")
    
    with st.expander("⚙️ PUE & Load Dynamics", expanded=False):
        # Defaults inteligentes por tipo
        pue_defaults = {"AI Factory (Training)": 1.15, "AI Factory (Inference)": 1.20, "Hyperscale Standard": 1.25}
        def_pue = pue_defaults.get(dc_type, 1.4)
        
        pue = st.slider("Design PUE", 1.05, 2.00, def_pue, 0.05)
        
        # Alias para compatibilidad con motor
        pue_input = pue 
        
        c_dyn1, c_dyn2 = st.columns(2)
        load_step_pct = c_dyn1.number_input("Max Step (%)", 0.0, 100.0, 40.0 if "AI" in dc_type else 20.0)
        
        # Alias para compatibilidad
        step_load_req = load_step_pct 
        
        spinning_res_pct = c_dyn2.number_input("Spinning Res (%)", 0.0, 100.0, 20.0)
        
        # Variables de pérdidas requeridas por el motor (Defaults ocultos)
        dist_loss_pct = 0.015
        gen_parasitic_pct = 0.025
        
        # Perfil Anual
        st.markdown("**Annual Profile:**")
        capacity_factor = st.slider("Capacity Factor", 0.3, 1.0, 0.90)
        peak_avg_ratio = st.slider("Peak/Avg Ratio", 1.0, 2.0, 1.15)
        load_ramp_req = st.number_input("Ramp Req (MW/s)", 0.1, 10.0, 3.0 if "AI" in dc_type else 1.0)

    # Cálculos intermedios para visualización
    p_total_dc = p_it * pue
    p_total_avg = p_total_dc * capacity_factor
    p_total_peak = p_total_dc * peak_avg_ratio
    
    st.info(f"Avg Load: **{p_total_avg:.1f} MW** | Peak: **{p_total_peak:.1f} MW**")

    # -------------------------------------------------------------------------
    # 2. SITIO (SITE CONDITIONS)
    # -------------------------------------------------------------------------
    st.header("2. Site Conditions")
    
    with st.expander("🌍 Ambient & Constraints", expanded=True):
        derate_mode = st.radio("Derate Mode", ["Auto-Calculate", "Manual"], horizontal=True, label_visibility="collapsed")
        
        if derate_mode == "Auto-Calculate":
            c_site1, c_site2 = st.columns(2)
            if is_imperial:
                site_temp_f = c_site1.number_input(f"Temp ({u_temp})", 32, 130, 95)
                site_alt_ft = c_site2.number_input(f"Alt ({u_dist})", 0, 10000, 300)
                site_temp_c = (site_temp_f - 32) * 5/9
                site_alt_m = site_alt_ft * 0.3048
            else:
                site_temp_c = c_site1.number_input(f"Temp ({u_temp})", 0, 55, 35)
                site_alt_m = c_site2.number_input(f"Alt ({u_dist})", 0, 3000, 100)
            
            methane_number = st.slider("Gas Methane Number (MN)", 30, 100, 80)
            
            # Cálculo de Derateo
            temp_derate = 1.0 - max(0, (site_temp_c - 25) * 0.01)
            alt_derate = 1.0 - max(0, (site_alt_m - 100) * 0.0001)
            fuel_derate = 1.0 if methane_number >= 70 else 0.95
            derate_factor_calc = max(0.1, temp_derate * alt_derate * fuel_derate)
        else:
            derate_factor_calc = st.slider("Manual Derate Factor", 0.1, 1.0, 0.9)
            site_temp_c = 25; site_alt_m = 0; methane_number = 80 # Defaults
            
        st.caption(f"📉 **Site Factor: {derate_factor_calc:.1%}**")
        
        # Area Constraint
        enable_footprint_limit = st.checkbox("Limit Area?")
        max_area_m2 = 999999999
        if enable_footprint_limit:
            max_area_input = st.number_input(f"Max Area ({u_area})", 100, 100000, 10000)
            if is_imperial: max_area_m2 = max_area_input / 10.764
            else: max_area_m2 = max_area_input

    # -------------------------------------------------------------------------
    # 3. TECNOLOGÍA (TECH)
    # -------------------------------------------------------------------------
    st.header("3. Technology")
    
    gen_filter = st.multiselect("Tech Filter", ["High Speed", "Medium Speed", "Gas Turbine"], default=["High Speed", "Medium Speed"])
    
    with st.expander("🔋 BESS & Options", expanded=False):
        use_bess = st.checkbox("Include BESS", value=("AI" in dc_type))
        
        # --- CORRECCIÓN: Definir variables por defecto para evitar NameError ---
        bess_strategy = "Transient Only"
        bess_reliability_enabled = False 
        
        if use_bess:
            bess_strategy = st.radio("Strategy", ["Transient Only", "Hybrid (Balanced)", "Reliability Priority"], index=1)
            # Definimos la variable crítica aquí
            bess_reliability_enabled = bess_strategy != "Transient Only"
            
        enable_black_start = st.checkbox("Black Start", value=True)
        include_chp = st.checkbox("Include Tri-Gen (CHP)", value=False)
        cooling_method = st.selectbox("Cooling", ["Air-Cooled", "Water-Cooled"])
        
    # Fuel Configuration
        st.markdown("---")
        st.markdown("⛽ **Fuel Strategy**")
        fuel_mode = st.radio("Fuel Source", ["Pipeline Gas", "LNG (Virtual Pipeline)", "Dual-Fuel (Pipe + LNG Backup)"])
        
        is_lng_primary = "LNG" in fuel_mode
        has_lng_storage = fuel_mode in ["LNG (Virtual Pipeline)", "Dual-Fuel (Pipe + LNG Backup)"]
        
        lng_days = 0
        dist_gas_main_m = 1000
        
        if has_lng_storage:
            lng_days = st.slider("On-Site Fuel Autonomy (Days)", 1, 15, 5, help="Days of on-site storage required")
            
            if is_lng_primary:
                st.caption("🚛 **LNG Logistics:** Primary Fuel Source")
            else:
                st.caption("🛡️ **LNG Logistics:** Backup Only")
    
    # Voltaje
    with st.expander("⚡ Electrical"):
        volt_mode = st.radio("Voltage", ["Auto-Recommend", "Manual"], horizontal=True)
        manual_voltage_kv = 13.8
        if volt_mode == "Manual":
            manual_voltage_kv = st.number_input("KV", 0.4, 69.0, 13.8)    

    # --- NUEVO: PÉRDIDAS DE DISTRIBUCIÓN ---
    st.markdown("📉 **Distribution Losses**")
    dist_loss_pct = st.slider(
        "Transformer & Cable Losses (%)", 
        0.0, 5.0, 1.5, 
        step=0.1, 
        help="Losses from generator terminals to IT connection point."
    )
    
    # Ajustamos la carga requerida en bornes del generador
    # Si el DC necesita 100 MW y perdemos 2%, el Gen debe producir 100 / (1 - 0.02)
    p_avg_at_gen = p_total_avg / (1 - dist_loss_pct/100)
    p_peak_at_gen = p_total_peak / (1 - dist_loss_pct/100)
    
    if dist_loss_pct > 0:
        st.caption(f"⚠️ Gen Output Required: **{p_avg_at_gen:.1f} MW** (+{p_avg_at_gen - p_total_avg:.1f} MW losses)")
    
    # -------------------------------------------------------------------------
    # 4. ECONOMÍA (ECONOMICS)
    # -------------------------------------------------------------------------
    st.header("4. Economics")
    
    c_eco1, c_eco2 = st.columns(2)
    
    # Precio base (Gasoducto)
    gas_price_pipeline = c_eco1.number_input("Pipeline Gas ($/MMBtu)", 0.5, 20.0, 3.5, step=0.1)
    
    # Precio LNG (Si aplica)
    if has_lng_storage:
        gas_price_lng = c_eco2.number_input("LNG Delivered ($/MMBtu)", 4.0, 30.0, 9.5, step=0.5, help="Includes molecule, liquefaction, and freight.")
    else:
        gas_price_lng = 0.0
        
    # Definir precio final para OPEX según el modo
    if is_lng_primary:
        total_gas_price = gas_price_lng
        st.info(f"Using LNG Price: **${total_gas_price:.2f}/MMBtu**")
    elif fuel_mode == "Dual-Fuel (Pipe + LNG Backup)":
        # Asumimos 95% uso de ducto y 5% pruebas con LNG para el blended price
        total_gas_price = (gas_price_pipeline * 0.95) + (gas_price_lng * 0.05)
        st.caption(f"Blended Price: **${total_gas_price:.2f}/MMBtu**")
    else:
        total_gas_price = gas_price_pipeline
    
    benchmark_price = st.number_input("Grid Price ($/kWh)", 0.05, 0.50, 0.12)
    
    # ... (después de benchmark_price) ...
    
    # --- NUEVO: BESS ECONOMICS (Editable) ---
    # --- NUEVO: BESS ECONOMICS (Editable) ---
    with st.expander("🔋 BESS Economics", expanded=False):
        c_bess1, c_bess2 = st.columns(2)
        bess_cost_kw = c_bess1.number_input("BESS Power CAPEX ($/kW)", 100.0, 1000.0, 250.0, step=10.0, help="Inverter & PCS cost")
        bess_cost_kwh = c_bess2.number_input("BESS Energy CAPEX ($/kWh)", 100.0, 1000.0, 400.0, step=10.0, help="Battery rack cost (DC block)")
        
        c_bess3, c_bess4 = st.columns(2)
        bess_om_kw_yr = c_bess3.number_input("BESS O&M ($/kW-yr)", 0.0, 50.0, 5.0, step=0.5, help="Fixed annual maintenance")
        bess_life_batt = c_bess4.number_input("Battery Life (Yrs)", 5, 20, 10, help="Cell replacement interval (Augmentation)")
        
        bess_life_inv = st.number_input("Inverter Life (Yrs)", 5, 30, 15, help="Power electronics (PCS) useful life")
        
    # --- NUEVO: FUEL INFRASTRUCTURE (Editable) ---
    # --- NUEVO: FUEL INFRASTRUCTURE (Editable) ---
    with st.expander("⛽ Fuel Infra Economics", expanded=False):
        c_fuel1, c_fuel2 = st.columns(2)
        # Multiplicador general
        fuel_infra_mult = c_fuel1.number_input("Infra Cost Multiplier", 0.5, 3.0, 1.0, step=0.1, help="Adjustment factor for local costs (Civil works, installation)")
        
        # Costo específico del tanque
        if has_lng_storage:
            lng_tank_cost = c_fuel2.number_input("LNG Tank Cost ($)", 200000, 1000000, 450000, step=25000, help="Unit cost per cryogenic tank (60k gal)")
        else:
            lng_tank_cost = 450000
            
        if not is_lng_primary:
             dist_gas_main_m = c_fuel2.number_input("Dist. to Main (m)", 0, 50000, 1000, step=100)

    # --- NUEVO: EMISSIONS CONTROL ECONOMICS (Editable) ---
    with st.expander("💨 Emissions Control Economics", expanded=False):
        # Opción para forzar el sistema aunque el algoritmo diga que no hace falta
        force_emissions = st.checkbox("Force Aftertreatment System", value=False, help="Include SCR/Oxicat regardless of the calculated emissions level")
        
        c_em1, c_em2 = st.columns(2)
        cost_scr_kw = c_em1.number_input("SCR Cost ($/kW)", 0.0, 500.0, 75.0, step=5.0, help="Selective Catalytic Reduction (NOx)")
        cost_oxicat_kw = c_em2.number_input("OxiCat Cost ($/kW)", 0.0, 200.0, 25.0, step=5.0, help="Oxidation Catalyst (CO/VOC)")
    
    with st.expander("💰 Financial Specs", expanded=False):
        wacc = st.number_input("WACC (%)", 1.0, 15.0, 8.0) / 100
        project_years = st.number_input("Years", 5, 30, 20)
        
        enable_itc = st.checkbox("ITC (30%)", value=include_chp)
        enable_ptc = st.checkbox("PTC", value=False)
        enable_depreciation = st.checkbox("MACRS", value=True)
        
        region = st.selectbox("Region", ["US - Gulf Coast", "Europe - Western", "Latin America", "Asia Pacific"])
        regional_multipliers = {"US - Gulf Coast": 1.0, "Europe - Western": 1.35, "Latin America": 0.95, "Asia Pacific": 0.85}
        regional_mult = regional_multipliers.get(region, 1.0)
        
        carbon_price_per_ton = st.number_input("Carbon Price ($/ton)", 0, 200, 0)
        
        enable_lcoe_target = st.checkbox("Target LCOE")
        target_lcoe = 0.08
        if enable_lcoe_target:
            target_lcoe = st.number_input("Target ($/kWh)", 0.05, 0.30, 0.08)

# ==============================================================================
# 3. GENERATOR SELECTION & FLEET OPTIMIZATION
# ==============================================================================

available_gens = {k: v for k, v in leps_gas_library.items() if v["type"] in gen_filter}

if not available_gens:
    st.error("⚠️ No generators match filter. Adjust technology selection.")
    st.stop()

# Auto-select best generator
best_gen = None
best_score = -999

for gen_name, gen_data in available_gens.items():
    unit_derated = gen_data["iso_rating_mw"] * derate_factor_calc
    
    if unit_derated < (p_total_peak * 0.1):
        continue
    
    # Usamos load_step_pct (Golpe Físico) para ver si el motor aguanta
    step_match = 1.0 if gen_data["step_load_pct"] >= load_step_pct else 0.5
    eff_score = gen_data["electrical_efficiency"] * 10
    cost_score = -gen_data["est_cost_kw"] / 100
    density_score = gen_data["power_density_mw_per_m2"] * 20  # Favor high density
    
    total_score = step_match * 100 + eff_score + cost_score + density_score
    
    if total_score > best_score:
        best_score = total_score
        best_gen = gen_name

selected_gen = st.sidebar.selectbox(
    "🔧 Selected Generator",
    list(available_gens.keys()),
    index=list(available_gens.keys()).index(best_gen) if best_gen else 0
)

gen_data = available_gens[selected_gen]

# ============================================================================
# GENERATOR PARAMETERS - EDITABLE (NEW FEATURE)
# ============================================================================
with st.sidebar.expander("⚙️ Generator Parameters (Editable)", expanded=False):
    st.markdown("**Capacity & Electrical:**")
    
    # 1. POTENCIA (ISO Rating)
    iso_mw_edit = st.number_input(
        "ISO Rating (MW)",
        value=float(gen_data["iso_rating_mw"]), # Asegurar float
        min_value=0.5, max_value=100.0, step=0.1,
        help="Generator gross prime rating (ISO)"
    )
    gen_data["iso_rating_mw"] = iso_mw_edit
    
    # 2. TENSIÓN (Voltage)
    gen_voltage = st.number_input(
        "Gen Terminal Voltage (kV)",
        value=13.8, # Valor típico
        step=0.1,
        help="Alternator output voltage"
    )
    
    # 3. CARGAS AUXILIARES (Parasitic Load)
    gen_aux_pct = st.number_input(
        "Gen Auxiliaries / Parasitic (%)",
        value=2.0, # Default típico (ventiladores, bombas, etc.)
        min_value=0.0, max_value=10.0, step=0.1,
        help="Generator parasitic consumption (radiators, pumps, ventilation)"
    )
    gen_data["aux_pct"] = gen_aux_pct # Guardamos en el diccionario
    
    st.markdown("---")
    st.markdown("**Reliability & Maintenance:**")
    # ... (El resto del código de mantenimiento sigue igual) ..
    
    # MTBF (Mean Time Between Failures)
    mtbf_edit = st.number_input(
        "MTBF (hours)",
        value=gen_data["mtbf_hours"],
        min_value=10000,
        max_value=150000,
        step=1000,
        help="Mean Time Between Failures - affects unplanned outages"
    )
    gen_data["mtbf_hours"] = mtbf_edit
    
    # Maintenance Interval
    maint_interval_edit = st.number_input(
        "Maintenance Interval (hrs)",
        value=gen_data["maintenance_interval_hrs"],
        min_value=500,
        max_value=20000,
        step=100,
        help="Hours between planned maintenance events"
    )
    gen_data["maintenance_interval_hrs"] = maint_interval_edit
    
    # Maintenance Duration
    maint_duration_edit = st.number_input(
        "Maintenance Duration (hrs)",
        value=gen_data["maintenance_duration_hrs"],
        min_value=12,
        max_value=240,
        step=6,
        help="Downtime per maintenance event"
    )
    gen_data["maintenance_duration_hrs"] = maint_duration_edit
    
    # Calculate and show availability impact
    annual_maint_hrs = (8760 / maint_interval_edit) * maint_duration_edit
    unit_avail = mtbf_edit / (mtbf_edit + 48 + annual_maint_hrs)
    
    st.markdown("---")
    st.markdown("**Calculated Unit Availability:**")
    st.metric("Single Unit", f"{unit_avail*100:.2f}%")
    
    maint_unavail = (annual_maint_hrs / 8760) * 100
    failure_unavail = (48 / (mtbf_edit + 48)) * 100
    
    st.caption(f"📊 Breakdown:")
    st.caption(f"  • Planned Maint: {maint_unavail:.2f}% unavailable")
    st.caption(f"  • Failures (MTTR=48h): {failure_unavail:.3f}% unavailable")
    st.caption(f"  • **Total: {((1-unit_avail)*100):.2f}% unavailable**")
    
    st.markdown("---")
    st.markdown("**Performance:**")
    
    # Efficiency
    eff_edit = st.number_input(
        "Electrical Efficiency",
        value=gen_data["electrical_efficiency"],
        min_value=0.25,
        max_value=0.60,
        step=0.001,
        format="%.3f",
        help="Electrical efficiency (HHV basis)"
    )
    gen_data["electrical_efficiency"] = eff_edit
    
    # Step Load Capability
    step_edit = st.number_input(
        "Step Load Capability (%)",
        value=gen_data["step_load_pct"],
        min_value=0.0,
        max_value=100.0,
        step=5.0,
        help="Maximum % load that can be accepted in one step"
    )
    gen_data["step_load_pct"] = step_edit
    
    # Ramp Rate
    ramp_edit = st.number_input(
        "Ramp Rate (MW/s)",
        value=gen_data["ramp_rate_mw_s"],
        min_value=0.1,
        max_value=5.0,
        step=0.1,
        help="Rate of load change capability"
    )
    gen_data["ramp_rate_mw_s"] = ramp_edit
    
    st.success("✅ Custom parameters applied")

# Derated capacity
unit_iso_cap = gen_data["iso_rating_mw"]
unit_site_cap = unit_iso_cap * derate_factor_calc

# ============================================================================
# CORRECTED: SPINNING RESERVE CALCULATION FOR Load/Unit (%)
# ============================================================================

# ============================================================================
# LOGICA SEPARADA: BESS (GOLPE) vs GENERADORES (RESERVA)
# ============================================================================

# ============================================================================
# LOGICA SEPARADA: BESS (GOLPE) vs GENERADORES (RESERVA) - ACTUALIZADO
# ============================================================================

# 1. Calcular BESS usando CARGA EN BORNES (Carga + Pérdidas)
bess_power_transient = 0.0
bess_energy_transient = 0.0
bess_breakdown_transient = {}

if use_bess:
    bess_power_transient, bess_energy_transient, bess_breakdown_transient = calculate_bess_requirements(
        p_avg_at_gen, p_peak_at_gen,  # <--- CAMBIO: Usamos carga con pérdidas
        load_step_pct,  
        gen_data["ramp_rate_mw_s"], gen_data["step_load_pct"],
        load_ramp_req,
        enable_black_start
    )

# 2. Calcular Generadores usando la POLÍTICA DE RESERVA y CARGA EN BORNES
spinning_reserve_result = calculate_spinning_reserve_units(
    p_avg_load=p_avg_at_gen,  # <--- CAMBIO: Usamos carga con pérdidas
    unit_capacity=unit_site_cap,
    spinning_reserve_pct=spinning_res_pct, 
    use_bess=use_bess,
    bess_power_mw=bess_power_transient if use_bess else 0,
    gen_step_capability_pct=gen_data["step_load_pct"]
)
# Extract results
n_running_for_spinning = spinning_reserve_result['n_units_running']
load_per_unit_spinning = spinning_reserve_result['load_per_unit_pct']
spinning_reserve_mw = spinning_reserve_result['spinning_reserve_mw']
spinning_from_gens = spinning_reserve_result['spinning_from_gens']
spinning_from_bess = spinning_reserve_result['spinning_from_bess']

# ============================================================================
# ENHANCED FLEET OPTIMIZATION - AVAILABILITY-DRIVEN WITH BESS CREDIT
# ============================================================================

# Step 1: Optimización de Flota usando CARGA EN BORNES
n_running_from_load, fleet_options = optimize_fleet_size(
    p_avg_at_gen, p_peak_at_gen, # <--- CAMBIO: Optimizar flota para carga con pérdidas
    unit_site_cap, load_step_pct, gen_data, use_bess
)

# Use the MAXIMUM of spinning reserve calculation and optimization
n_running_from_load = max(n_running_from_load, n_running_for_spinning)

# Step 2: Calculate N+X for availability target
avail_decimal = avail_req / 100
mtbf_hours = gen_data["mtbf_hours"]
mttr_hours = 48  # Realistic: 2 days repair time

# ============================================================================
# HYBRID ALGORITHM: Generate comparison table of Gen+BESS configurations
# ============================================================================

reliability_configs = []

# Configuration A: No BESS (Baseline)
# Calculate spinning reserve requirements WITHOUT BESS
spinning_no_bess = calculate_spinning_reserve_units(
    p_avg_load=p_avg_at_gen,
    unit_capacity=unit_site_cap,
    spinning_reserve_pct=spinning_res_pct,  # <--- CORREGIDO (Política)
    use_bess=False,
    bess_power_mw=0,
    gen_step_capability_pct=gen_data["step_load_pct"]
)

# Config A: Use the calculated n_running directly from spinning reserve function
# FIX: Ensure we also cover PEAK load (which might be higher than Avg + Spinning)
n_running_peak = math.ceil(p_total_peak / unit_site_cap)
n_running_no_bess = max(spinning_no_bess['n_units_running'], n_running_peak)

# Recalculate load pct based on the final n_running (Critical step!)
load_pct_no_bess = (p_avg_at_gen / (n_running_no_bess * unit_site_cap)) * 100

print(f"[DEBUG] Config A: n_running={n_running_no_bess} (Peak req: {n_running_peak}), load={load_pct_no_bess:.1f}%", file=sys.stderr)

# Find minimum reserve units (N+X) needed for availability target
best_config_a = None
for n_res in range(0, 100):
    n_tot = n_running_no_bess + n_res
    
    avg_avail, _ = calculate_availability_weibull(
        n_tot, n_running_no_bess, mtbf_hours, project_years, 
        gen_data["maintenance_interval_hrs"], gen_data["maintenance_duration_hrs"]
    )
    
    if avg_avail >= avail_decimal:
        # Calculate efficiency at this load point
        eff_a = get_part_load_efficiency(gen_data["electrical_efficiency"], load_pct_no_bess, gen_data["type"])
        
        print(f"[DEBUG] Config A FOUND: n_run={n_running_no_bess}, n_res={n_res}, total={n_tot}, load={load_pct_no_bess:.1f}%, eff={eff_a*100:.2f}%, avail={avg_avail*100:.4f}%", file=sys.stderr)
        
        best_config_a = {
            'name': 'A: No BESS',
            'n_running': n_running_no_bess,
            'n_reserve': n_res,
            'n_total': n_tot,
            'bess_mw': 0,
            'bess_mwh': 0,
            'bess_credit': 0,
            'availability': avg_avail,
            'load_pct': load_pct_no_bess,
            'efficiency': eff_a,
            'spinning_reserve_mw': spinning_no_bess['spinning_reserve_mw'],
            'spinning_from_gens': spinning_no_bess['spinning_from_gens'],
            'spinning_from_bess': 0,
            'headroom_mw': spinning_no_bess['headroom_available']
        }
        break

# Fallback if availability target not met
if not best_config_a:
    n_res_fallback = 100
    n_tot_fallback = n_running_no_bess + n_res_fallback
    
    fallback_avail, _ = calculate_availability_weibull(
        n_tot_fallback, n_running_no_bess, mtbf_hours, project_years,
        gen_data["maintenance_interval_hrs"], gen_data["maintenance_duration_hrs"]
    )
    eff_a = get_part_load_efficiency(gen_data["electrical_efficiency"], load_pct_no_bess, gen_data["type"])
    
    print(f"[DEBUG] Config A FALLBACK: n_run={n_running_no_bess}, n_res={n_res_fallback}", file=sys.stderr)
    
    best_config_a = {
        'name': 'A: No BESS',
        'n_running': n_running_no_bess,
        'n_reserve': n_res_fallback,
        'n_total': n_tot_fallback,
        'bess_mw': 0,
        'bess_mwh': 0,
        'bess_credit': 0,
        'availability': fallback_avail,
        'load_pct': load_pct_no_bess,
        'efficiency': eff_a,
        'spinning_reserve_mw': spinning_no_bess['spinning_reserve_mw'],
        'spinning_from_gens': spinning_no_bess['spinning_from_gens'],
        'spinning_from_bess': 0,
        'headroom_mw': spinning_no_bess['headroom_available']
    }


reliability_configs.append(best_config_a)

# Configuration B: BESS Transient Only
if use_bess:
   # Calculate spinning reserve with BESS
    spinning_with_bess = calculate_spinning_reserve_units(
        p_avg_load=p_avg_at_gen,
        unit_capacity=unit_site_cap,
        spinning_reserve_pct=spinning_res_pct,  # <--- CORREGIDO (Política)
        use_bess=True,
        bess_power_mw=bess_power_transient,
        gen_step_capability_pct=gen_data["step_load_pct"]
    )
    
    # Use the calculated values directly
    n_running_with_bess = spinning_with_bess['n_units_running']
    load_pct_with_bess = spinning_with_bess['load_per_unit_pct']
    
    print(f"[DEBUG] Config B: n_running={n_running_with_bess}, load={load_pct_with_bess:.1f}%, BESS covers {spinning_with_bess['spinning_from_bess']:.1f} MW of spinning reserve", file=sys.stderr)
    
    # Find minimum reserve units for availability
    best_config_b = None
    for n_res in range(0, 100):
        n_tot = n_running_with_bess + n_res
        
        avg_avail, _ = calculate_availability_weibull(
            n_tot, n_running_with_bess, mtbf_hours, project_years,
            gen_data["maintenance_interval_hrs"], gen_data["maintenance_duration_hrs"]
        )
        
        if avg_avail >= avail_decimal:
            eff_b = get_part_load_efficiency(gen_data["electrical_efficiency"], load_pct_with_bess, gen_data["type"])
            
            print(f"[DEBUG] Config B FOUND: n_run={n_running_with_bess}, n_res={n_res}, total={n_tot}, load={load_pct_with_bess:.1f}%, eff={eff_b*100:.2f}%", file=sys.stderr)
            
            best_config_b = {
                'name': 'B: BESS Transient',
                'n_running': n_running_with_bess,
                'n_reserve': n_res,
                'n_total': n_tot,
                'bess_mw': bess_power_transient,
                'bess_mwh': bess_energy_transient,
                'bess_credit': 0,
                'availability': avg_avail,
                'load_pct': load_pct_with_bess,
                'efficiency': eff_b,
                'spinning_reserve_mw': spinning_with_bess['spinning_reserve_mw'],
                'spinning_from_gens': spinning_with_bess['spinning_from_gens'],
                'spinning_from_bess': spinning_with_bess['spinning_from_bess'],
                'headroom_mw': spinning_with_bess['headroom_available']
            }
            break
    
    # Fallback
    if not best_config_b:
        n_res_fallback = 100
        n_tot_fallback = n_running_with_bess + n_res_fallback
        fallback_avail_b, _ = calculate_availability_weibull(
            n_tot_fallback, n_running_with_bess, mtbf_hours, project_years,
            gen_data["maintenance_interval_hrs"], gen_data["maintenance_duration_hrs"]
        )
        eff_b = get_part_load_efficiency(gen_data["electrical_efficiency"], load_pct_with_bess, gen_data["type"])
        
        best_config_b = {
            'name': 'B: BESS Transient',
            'n_running': n_running_with_bess,
            'n_reserve': n_res_fallback,
            'n_total': n_tot_fallback,
            'bess_mw': bess_power_transient,
            'bess_mwh': bess_energy_transient,
            'bess_credit': 0,
            'availability': fallback_avail_b,
            'load_pct': load_pct_with_bess,
            'efficiency': eff_b,
            'spinning_reserve_mw': spinning_with_bess['spinning_reserve_mw'],
            'spinning_from_gens': spinning_with_bess['spinning_from_gens'],
            'spinning_from_bess': spinning_with_bess['spinning_from_bess'],
            'headroom_mw': spinning_with_bess['headroom_available']
        }
    
    reliability_configs.append(best_config_b)

# Configuration C: BESS Hybrid (Balanced)
if use_bess and bess_reliability_enabled:
    print(f"[DEBUG] Config C: Starting calculation", file=sys.stderr)
    
    # Start with same capacity-based sizing as Config B
    n_running_min_c = spinning_with_bess['n_units_running']
    n_running_optimal_c = n_running_min_c
    
    print(f"[DEBUG] Config C: n_running for avg+spinning: {n_running_optimal_c}", file=sys.stderr)
    
    # BESS sizing for reliability
    if bess_strategy == 'Hybrid (Balanced)':
        target_gensets_covered = 5  # Increased from 3
        bess_coverage_hrs = 2.0
    else:  # Reliability Priority
        target_gensets_covered = 8  # Increased from 5
        bess_coverage_hrs = 2.5
    
    # Size BESS
    bess_power_hybrid = max(
        bess_power_transient,
        target_gensets_covered * unit_site_cap
    )
    bess_energy_hybrid = bess_power_hybrid * bess_coverage_hrs
    min_energy = target_gensets_covered * unit_site_cap * bess_coverage_hrs
    bess_energy_hybrid = max(bess_energy_hybrid, min_energy)
    
    # Calculate BESS credit
    try:
        bess_credit_units, credit_breakdown = calculate_bess_reliability_credit(
            bess_power_hybrid, bess_energy_hybrid, unit_site_cap, mttr_hours
        )
        
        # Apply 65% de-rating for conservatism (was 50%)
        bess_credit_conservative = bess_credit_units * 0.65
        bess_credit_int = max(0, int(bess_credit_conservative))
        
        print(f"[DEBUG] Config C: BESS Power={bess_power_hybrid:.1f} MW, Energy={bess_energy_hybrid:.1f} MWh", file=sys.stderr)
        print(f"[DEBUG] Config C: BESS Credit raw={bess_credit_units:.2f}, conservative={bess_credit_conservative:.2f}, int={bess_credit_int}", file=sys.stderr)
        
        # Show debug info
        st.sidebar.markdown("---")
        st.sidebar.markdown("🔍 **BESS Credit Debug**")
        st.sidebar.caption(f"Power: {bess_power_hybrid:.1f} MW")
        st.sidebar.caption(f"Energy: {bess_energy_hybrid:.1f} MWh")
        st.sidebar.caption(f"Raw credit: {bess_credit_units:.2f}")
        st.sidebar.caption(f"Applied: {bess_credit_int} units")
        
    except Exception as e:
        st.sidebar.error(f"BESS credit calc error: {str(e)}")
        bess_credit_int = 0
        bess_credit_conservative = 0
        credit_breakdown = {}
    
    best_config_c = None
    found_c = False
    
    # Strategy: Start with Config B's reserve requirement, then apply BESS credit
    # First, find what reserve Config B needed
    if best_config_b and 'n_reserve' in best_config_b:
        config_b_reserve = best_config_b['n_reserve']
    else:
        # Fallback: estimate from Config A or use default
        config_b_reserve = best_config_a['n_reserve'] if best_config_a and 'n_reserve' in best_config_a else 8
    
    print(f"[DEBUG] Config C: Config B used n_reserve={config_b_reserve}, will try LOWER reserve with BESS backup", file=sys.stderr)
    
    # Search for best config - try with LESS reserve than Config B
    for n_run_offset in range(-5, 10):
        if found_c:
            break
        
        n_run = n_running_optimal_c + n_run_offset
        if n_run < n_running_min_c:
            continue
        
        if n_run * unit_site_cap < spinning_with_bess['required_online_capacity']:
            continue
        
        # NEW STRATEGY: Try lower reserve values (0 to config_b_reserve)
        for n_res_try in range(max(1, config_b_reserve - 10), config_b_reserve + 2):
            n_tot = n_run + n_res_try
            
            print(f"[DEBUG] Config C: Trying n_run={n_run}, n_res={n_res_try}, total={n_tot} (vs Config B's {config_b_reserve})", file=sys.stderr)
            
            try:
                avg_avail, _ = calculate_availability_weibull(
                    n_tot, n_run, mtbf_hours, project_years,
                    gen_data["maintenance_interval_hrs"],
                    gen_data["maintenance_duration_hrs"]
                )
                
                # BESS provides additional reliability margin
                bess_reliability_boost = min(0.0005, bess_credit_int * 0.00005)
                avg_avail_with_bess = min(1.0, avg_avail + bess_reliability_boost)
                
                print(f"[DEBUG] Config C: Avail without BESS={avg_avail*100:.4f}%, with BESS boost={avg_avail_with_bess*100:.4f}%", file=sys.stderr)
                
                if avg_avail_with_bess >= avail_decimal:
                    load_pct_c = (p_avg_at_gen / (n_run * unit_site_cap)) * 100
                    eff_c = get_part_load_efficiency(
                        gen_data["electrical_efficiency"],
                        load_pct_c,
                        gen_data["type"]
                    )
                    
                    print(f"[DEBUG] Config C FOUND: n_run={n_run}, n_res={n_res_try}, total={n_tot}, avail={avg_avail_with_bess*100:.4f}%", file=sys.stderr)
                    
                    best_config_c = {
                        'name': f'C: {bess_strategy}',
                        'n_running': n_run,
                        'n_reserve': n_res_try,
                        'n_total': n_tot,
                        'bess_mw': bess_power_hybrid,
                        'bess_mwh': bess_energy_hybrid,
                        'bess_credit': bess_credit_conservative,
                        'availability': avg_avail_with_bess,
                        'credit_breakdown': credit_breakdown,
                        'load_pct': load_pct_c,
                        'efficiency': eff_c,
                        'score': eff_c,
                        'spinning_reserve_mw': spinning_with_bess['spinning_reserve_mw'],
                        'spinning_from_gens': spinning_with_bess['spinning_from_gens'],
                        'spinning_from_bess': spinning_with_bess['spinning_from_bess'],
                        'headroom_mw': n_run * unit_site_cap - p_total_avg
                    }
                    found_c = True
                    break
            except Exception as e:
                print(f"[DEBUG] Config C error: {str(e)}", file=sys.stderr)
                continue
    
    if best_config_c:
        reliability_configs.append(best_config_c)
    else:
        # Fallback
        st.sidebar.warning("⚠️ Config C: Using fallback")
        n_run_fallback = n_running_min_c
        n_res_fallback = 100
        
        fallback_avail_c, _ = calculate_availability_weibull(
            n_run_fallback + n_res_fallback, n_run_fallback,
            mtbf_hours, project_years,
            gen_data["maintenance_interval_hrs"],
            gen_data["maintenance_duration_hrs"]
        )
        
        fallback_load_c = (p_total_avg / (n_run_fallback * unit_site_cap)) * 100
        
        best_config_c = {
            'name': f'C: {bess_strategy} (fallback)',
            'n_running': n_run_fallback,
            'n_reserve': n_res_fallback,
            'n_total': n_run_fallback + n_res_fallback,
            'bess_mw': bess_power_hybrid,
            'bess_mwh': bess_energy_hybrid,
            'bess_credit': bess_credit_conservative,
            'availability': fallback_avail_c,
            'load_pct': fallback_load_c,
            'spinning_reserve_mw': spinning_with_bess['spinning_reserve_mw'],
            'spinning_from_gens': spinning_with_bess['spinning_from_gens'],
            'spinning_from_bess': spinning_with_bess['spinning_from_bess'],
            'headroom_mw': n_run_fallback * unit_site_cap - p_total_avg
        }
        reliability_configs.append(best_config_c)

# Select final configuration based on strategy
if bess_strategy == "Transient Only" and len(reliability_configs) >= 2:
    selected_config = reliability_configs[1]  # Config B
elif bess_strategy in ["Hybrid (Balanced)", "Reliability Priority"] and len(reliability_configs) >= 3:
    selected_config = reliability_configs[2]  # Config C
elif len(reliability_configs) >= 1:
    selected_config = reliability_configs[0]  # Config A (fallback)
else:
    # Emergency fallback
    selected_config = {
        'name': 'Fallback',
        'n_running': n_running_from_load,
        'n_reserve': 10,
        'n_total': n_running_from_load + 10,
        'bess_mw': bess_power_transient if use_bess else 0,
        'bess_mwh': bess_energy_transient if use_bess else 0,
        'bess_credit': 0,
        'availability': 0.9999,
        'load_pct': (p_total_avg / (n_running_from_load * unit_site_cap)) * 100,
        'spinning_reserve_mw': spinning_reserve_mw,
        'spinning_from_gens': spinning_from_gens,
        'spinning_from_bess': spinning_from_bess
    }

# Extract final values
n_running = selected_config['n_running']
n_reserve = selected_config['n_reserve']
n_total = selected_config['n_total']
prob_gen = selected_config['availability']
bess_power_total = selected_config['bess_mw']
bess_energy_total = selected_config['bess_mwh']
target_met = prob_gen >= avail_decimal

# CRITICAL: Use the load_pct from the selected config (properly calculated with spinning reserve)
load_per_unit_pct = selected_config['load_pct']

# Update BESS breakdown
if use_bess and bess_power_total > 0:
    bess_breakdown = bess_breakdown_transient.copy()
    bess_breakdown['reliability_backup'] = bess_power_total - bess_power_transient
else:
    bess_breakdown = {}

installed_cap = n_total * unit_site_cap

# Calculate reliability curve
_, availability_curve = calculate_availability_weibull(n_total, n_running, mtbf_hours, project_years, gen_data["maintenance_interval_hrs"], gen_data["maintenance_duration_hrs"])

# Show optimization result in sidebar
if use_bess and bess_reliability_enabled and 'bess_credit' in selected_config:
    units_saved = best_config_a['n_total'] - selected_config['n_total']
    if units_saved > 0:
        st.sidebar.success(
            f"✅ **BESS Reliability Credit:**\n\n"
            f"Gensets: {best_config_a['n_total']} → {selected_config['n_total']} "
            f"({units_saved} units saved)\n\n"
            f"BESS Credit: {selected_config['bess_credit']:.1f} genset-equivalents"
        )
    else:
        st.sidebar.info(
            f"ℹ️ **{selected_config['name']}:**\n\n"
            f"BESS provides {selected_config['bess_credit']:.1f} reliability credit"
        )
elif not target_met:
    availability_gap = (avail_req - prob_gen * 100)
    st.sidebar.error(f"⚠️ **Availability Target Not Met**")
    st.sidebar.warning(
        f"Target: {avail_req:.3f}%\n\n"
        f"Achievable: {prob_gen*100:.3f}%\n\n"
        f"Gap: {availability_gap:.3f}%\n\n"
        f"Using configuration: N+{n_reserve}"
    )
    st.sidebar.info(
        "💡 **Recommendations:**\n"
        "- Use higher MTBF generators\n"
        "- Increase BESS capacity\n"
        "- Accept lower availability\n"
        "- Add grid backup connection"
    )

# Load Distribution Strategy
st.sidebar.markdown("⚡ **Load Distribution**")
load_strategy = st.sidebar.radio(
    "Operating Mode",
    ["Equal Loading (N units)", "Spinning Reserve (N+1)", "Sequential"],
    help="Load distribution strategy"
)

if load_strategy == "Equal Loading (N units)":
    units_running = n_running
elif load_strategy == "Spinning Reserve (N+1)":
    units_running = n_running + 1 if n_reserve > 0 else n_running
else:
    units_running = n_running

# ==============================================================================
# CORRECCIÓN DE EFICIENCIA POR SITIO (NUEVO PARCHE)
# ==============================================================================

# 1. Factor por Calidad de Gas (Methane Number)
# Si MN < 70, el motor retrasa el encendido (Timing), perdiendo eficiencia drásticamente.
if methane_number < 70:
    eff_fuel_factor = 0.94  # Pierde ~6% de eficiencia relativa
elif methane_number < 80:
    eff_fuel_factor = 0.98  # Pierde ~2%
else:
    eff_fuel_factor = 1.0

# 2. Factor por Altitud Extrema
# Sobre 2000m, el turbo trabaja fuera de rango óptimo, aumentando bombeo y calor.
if site_alt_m > 2000:
    # Pierde 0.5% relativo por cada 1000m adicionales sobre 2000m
    eff_alt_factor = 1.0 - ((site_alt_m - 2000) / 1000) * 0.005
else:
    eff_alt_factor = 1.0

# Factor Total de Corrección de Eficiencia
site_efficiency_correction = eff_fuel_factor * eff_alt_factor

# Cálculo Base (Interpolación de Carga - Curva ISO)
base_fleet_eff = get_part_load_efficiency(
    gen_data["electrical_efficiency"],
    load_per_unit_pct,
    gen_data["type"]
)

# Aplicar Corrección de Sitio a la eficiencia final
fleet_efficiency = base_fleet_eff * site_efficiency_correction

# ==============================================================================

# Voltage recommendation (adjusted for off-grid data centers)
if volt_mode == "Auto-Recommend":
    if installed_cap < 10:
        rec_voltage_kv = 4.16
    elif installed_cap < 50:
        rec_voltage_kv = 13.8
    elif installed_cap < 200:
        rec_voltage_kv = 34.5
    else:
        rec_voltage_kv = 34.5
else:
    rec_voltage_kv = manual_voltage_kv

# Transient stability (Check against the physical HIT, not the reserve)
stability_ok, voltage_sag = transient_stability_check(
    gen_data["reactance_xd_2"], units_running, load_step_pct
)

# ==============================================================================
# 4. FOOTPRINT CALCULATION & OPTIMIZATION
# ==============================================================================

# Calculate footprint per component
area_per_gen = 1 / gen_data["power_density_mw_per_m2"]
area_gen = n_total * unit_site_cap * area_per_gen

area_bess = bess_power_total * 30 if use_bess else 0

# LNG storage
if has_lng_storage:
    total_fuel_input_mw_temp = (p_total_avg / fleet_efficiency)
    total_fuel_input_mmbtu_hr_temp = total_fuel_input_mw_temp * 3.412
    lng_mmbtu_total = total_fuel_input_mmbtu_hr_temp * 24 * lng_days
    lng_gal = lng_mmbtu_total / 0.075
    storage_area_m2 = (lng_gal * 0.00378541) * 5
else:
    storage_area_m2 = 0
    lng_gal = 0

# Cooling/CHP
pue_base = 1.35 if cooling_method == "Water-Cooled" else 1.50
total_cooling_mw = p_it * (pue - 1.0)
area_chp = total_cooling_mw * 20 if include_chp else (p_total_avg * 10)

area_sub = 2500
total_area_m2 = (area_gen + storage_area_m2 + area_chp + area_bess + area_sub) * 1.2

# ==============================================================================
# FOOTPRINT OPTIMIZATION (LOGIC REPLACEMENT)
# ==============================================================================

# FOOTPRINT OPTIMIZATION
is_area_exceeded = total_area_m2 > max_area_m2
area_utilization_pct = (total_area_m2 / max_area_m2) * 100 if enable_footprint_limit else 0

footprint_recommendations = []

if is_area_exceeded and enable_footprint_limit:
    current_density = gen_data["power_density_mw_per_m2"]
    
    # -------------------------------------------------------------------------
    # STRATEGY 1: TECHNOLOGY SWITCH (Search FULL Library, ignore filters)
    # -------------------------------------------------------------------------
    # Buscamos en TODA la librería, no solo en lo seleccionado
    for alt_gen_name, alt_gen_data in leps_gas_library.items():
        # Solo sugerir si la densidad es mayor
        if alt_gen_data["power_density_mw_per_m2"] > current_density:
            
            # Recalcular fleet con la nueva tecnología
            alt_derate = derate_factor_calc # Asumimos mismo derate factor por simplicidad
            alt_unit_cap = alt_gen_data["iso_rating_mw"] * alt_derate
            
            # Fleet sizing
            if use_bess:
                alt_n_running = math.ceil(p_total_avg * 1.15 / alt_unit_cap)
            else:
                alt_n_running = math.ceil(p_total_peak / alt_unit_cap)
            
            # Mantener el mismo nivel de redundancia N+X original para la comparación justa
            alt_n_total = alt_n_running + n_reserve 
            
            # Calcular nueva área
            alt_area_per_gen = 1 / alt_gen_data["power_density_mw_per_m2"]
            alt_area_gen = alt_n_total * alt_unit_cap * alt_area_per_gen
            alt_total_area = (alt_area_gen + storage_area_m2 + area_chp + area_bess + area_sub) * 1.2
            
            # Si cabe (o mejora significativamente, digamos reduce el exceso a la mitad)
            if alt_total_area <= max_area_m2:
                savings_pct = ((total_area_m2 - alt_total_area) / total_area_m2) * 100
                
                footprint_recommendations.append({
                    'type': '🚀 Tech Switch',
                    'action': f'Switch to **{alt_gen_name}** ({alt_gen_data["type"]})',
                    'new_area': alt_total_area,
                    'savings_pct': savings_pct,
                    'trade_off': f'Higher Density ({alt_gen_data["power_density_mw_per_m2"]*1000:.0f} kW/m²)'
                })

    # -------------------------------------------------------------------------
    # STRATEGY 2: AGGRESSIVE REDUNDANCY REDUCTION (Iterative Loop)
    # -------------------------------------------------------------------------
    # Probamos reduciendo la reserva paso a paso hasta que quepa o lleguemos a N+0
    if n_reserve > 0:
        found_reduction = False
        
        # Iterar desde (Reserva - 1) hasta 0
        for r_try in range(n_reserve - 1, -1, -1):
            reduced_n_total = n_running + r_try
            
            # Recalcular área
            reduced_area_gen = reduced_n_total * unit_site_cap * area_per_gen
            reduced_total_area = (reduced_area_gen + storage_area_m2 + area_chp + area_bess + area_sub) * 1.2
            
            # Si logramos entrar en el área
            if reduced_total_area <= max_area_m2:
                # Calcular penalización de disponibilidad
                reduced_avail, _ = calculate_availability_weibull(
                    reduced_n_total, n_running, mtbf_hours, project_years, 
                    gen_data["maintenance_interval_hrs"], gen_data["maintenance_duration_hrs"]
                )
                
                # Si tenemos BESS, sumar el crédito
                if use_bess and 'bess_credit' in selected_config:
                     reduced_avail = min(0.9999, reduced_avail + 0.0005) # Approx boost
                
                savings_pct = ((total_area_m2 - reduced_total_area) / total_area_m2) * 100
                avail_drop = (prob_gen - reduced_avail) * 100
                
                footprint_recommendations.append({
                    'type': '📉 Reduce Redundancy',
                    'action': f'Reduce to **N+{r_try}** (Remove {n_reserve - r_try} units)',
                    'new_area': reduced_total_area,
                    'savings_pct': savings_pct,
                    'trade_off': f'⚠️ Availability drops to **{reduced_avail*100:.4f}%** (Target: {avail_req}%)'
                })
                found_reduction = True
                break # Encontramos la máxima reserva posible que cabe, paramos.
        
        # Si ni siquiera N+0 cabe, sugerir N+0 de todos modos como "Mejor Esfuerzo"
        if not found_reduction:
             r_try = 0
             reduced_n_total = n_running
             reduced_area_gen = reduced_n_total * unit_site_cap * area_per_gen
             reduced_total_area = (reduced_area_gen + storage_area_m2 + area_chp + area_bess + area_sub) * 1.2
             savings_pct = ((total_area_m2 - reduced_total_area) / total_area_m2) * 100
             
             footprint_recommendations.append({
                'type': '🚨 Extreme Reduction',
                'action': f'Minimize to **N+0** (No Redundancy)',
                'new_area': reduced_total_area,
                'savings_pct': savings_pct,
                'trade_off': f'⚠️ Critical Availability Risk. Still exceeds area by {(reduced_total_area/max_area_m2)*100 - 100:.1f}%'
            })

# Display conversions
if is_imperial:
    disp_area = total_area_m2 * 0.000247105
    disp_area_unit = "Acres"
else:
    disp_area = total_area_m2 / 10000
    disp_area_unit = "Ha"

# ==============================================================================
# 5. FUEL & EMISSIONS
# ==============================================================================

total_fuel_input_mw = (p_total_avg / fleet_efficiency)
total_fuel_input_mmbtu_hr = total_fuel_input_mw * 3.412

# --- LNG CALCULATION ENGINE (NUEVO) ---
lng_metrics = {}

if has_lng_storage:
    # 1. Factores de Conversión LNG
    # 1 MMBtu ≈ 12.0 - 12.5 Galones -> Usamos 0.0825 MMBtu/Gal
    mmbtu_per_gallon = 0.0825 
    truck_capacity_gal = 10000 # Camión estándar 10k galones
    
    # 2. Consumo Volumétrico
    consumption_gal_hr = total_fuel_input_mmbtu_hr / mmbtu_per_gallon
    consumption_gal_day = consumption_gal_hr * 24
    
    # 3. Dimensionamiento de Almacenamiento (Storage)
    required_storage_gal = consumption_gal_day * lng_days
    
    # Tanques comerciales típicos (60k galones es estándar industrial)
    tank_size_gal = 60000 
    num_tanks = math.ceil(required_storage_gal / tank_size_gal)
    # Mínimo 1 tanque si hay almacenamiento
    num_tanks = max(1, num_tanks)
    
    # 4. Logística de Camiones
    if is_lng_primary:
        trucks_per_day = consumption_gal_day / truck_capacity_gal
        trucks_per_week = trucks_per_day * 7
    else:
        # Si es backup, tráfico mínimo (boil-off replacement)
        trucks_per_day = 0 
        trucks_per_week = 0.5 
        
    lng_metrics = {
        "gal_per_day": consumption_gal_day,
        "storage_gal": required_storage_gal,
        "num_tanks": num_tanks,
        "tank_size": tank_size_gal,
        "trucks_day": trucks_per_day,
        "trucks_week": trucks_per_week,
        "peak_flow_mmbtu_hr": total_fuel_input_mmbtu_hr * (p_total_peak/p_total_avg)
    }

# Pipeline sizing (lógica actualizada)
if not is_lng_primary:
    flow_rate_scfh = total_fuel_input_mmbtu_hr * 1000 / 1.02
    rec_pipe_dia = math.sqrt(flow_rate_scfh / 3000) * 2
else:
    rec_pipe_dia = 0

# Emissions
nox_lb_hr = (p_total_avg * 1000) * (gen_data["emissions_nox"] / 1000)
co_lb_hr = (p_total_avg * 1000) * (gen_data["emissions_co"] / 1000)
co2_ton_yr = total_fuel_input_mmbtu_hr * 0.0531 * 8760 * capacity_factor

# Emissions control CAPEX Calculation
at_capex_total = 0

# --- CORRECCIÓN: Calcular horas efectivas aquí para evitar el NameError ---
effective_hours = 8760 * capacity_factor
# -------------------------------------------------------------------------

# Convertir libras/hora a Toneladas/año
nox_tons_year = (nox_lb_hr * effective_hours) / 2000 

# REGLA: Se instala si supera 100 Ton/año (Major Source) O si el usuario lo fuerza
if nox_tons_year > 100 or force_emissions:
    # Usamos los costos unitarios definidos en el Sidebar
    total_ats_cost_kw = cost_scr_kw + cost_oxicat_kw
    at_capex_total = (installed_cap * 1000) * total_ats_cost_kw
    
    if force_emissions:
        st.toast("⚠️ Emissions Control Forced by User")

# ==============================================================================
# 6. COOLING & TRI-GENERATION
# ==============================================================================

total_heat_rec_mw = 0.0
total_cooling_mw_chp = 0.0
cooling_coverage_pct = 0.0

if include_chp:
    waste_heat_mw = total_fuel_input_mw - p_total_avg
    recovery_eff = 0.65
    total_heat_rec_mw = waste_heat_mw * recovery_eff
    
    cop_absorption = 0.70
    total_cooling_mw_chp = total_heat_rec_mw * cop_absorption
    cooling_coverage_pct = min(100.0, (total_cooling_mw_chp / total_cooling_mw) * 100)
    
    pue_improvement = 0.15 * (cooling_coverage_pct / 100)
    pue_actual = pue - pue_improvement
else:
    pue_actual = pue

# Water consumption
wue = 1.8 if (cooling_method == "Water-Cooled" or include_chp) else 0.2
water_m3_day = p_it * wue * 24

if is_imperial:
    disp_cooling = total_cooling_mw_chp * 284.3
    disp_water = water_m3_day * 264.172
else:
    disp_cooling = total_cooling_mw_chp
    disp_water = water_m3_day

# ==============================================================================
# 7. ENHANCED FINANCIALS & LCOE
# ==============================================================================

# Apply regional multiplier
# --- NUEVO: ECONOMÍA DE ESCALA ---
# Penalizar proyectos pequeños (<20 MW) porque los costos fijos (ingeniería, permisos) pesan más.
scale_factor = 1.0
if installed_cap < 2.5: 
    scale_factor = 1.30  # +30% costo unitario para Microgrids (<2.5 MW)
elif installed_cap < 10.0:
    scale_factor = 1.15  # +15% costo unitario para Pequeños (<10 MW)
elif installed_cap < 50.0:
    scale_factor = 1.05  # +5% costo unitario (<50 MW)
# ---------------------------------

gen_unit_cost = gen_data["est_cost_kw"] * regional_mult * scale_factor
gen_install_cost = gen_data["est_install_kw"] * regional_mult * scale_factor

gen_cost_total = (installed_cap * 1000) * gen_unit_cost / 1e6

# Installation & BOP
idx_install = gen_install_cost / gen_unit_cost
idx_chp = 0.20 if include_chp else 0

# ... (código anterior: gen_install_cost = ...) ...

gen_cost_total = (installed_cap * 1000) * gen_unit_cost / 1e6

# Installation & BOP
idx_install = gen_install_cost / gen_unit_cost
idx_chp = 0.20 if include_chp else 0

# BESS costs (Variables now come from Sidebar Section 4)
# (Ya no hay líneas fijas aquí)

if use_bess:
    cost_power_part = (bess_power_total * 1000) * bess_cost_kw
    cost_energy_part = (bess_energy_total * 1000) * bess_cost_kwh
    # ... resto del cálculo igual ...

if use_bess:
    cost_power_part = (bess_power_total * 1000) * bess_cost_kw
    cost_energy_part = (bess_energy_total * 1000) * bess_cost_kwh
    bess_capex_m = (cost_power_part + cost_energy_part) / 1e6
    bess_om_annual = (bess_power_total * 1000 * bess_om_kw_yr)
else:
    bess_capex_m = 0
    bess_om_annual = 0

# --- LNG INFRASTRUCTURE CAPEX (DETALLADO & EDITABLE) ---
lng_capex_m = 0.0
pipeline_capex_m = 0.0

if has_lng_storage:
    # 1. Costos Unitarios (Ajustados por Multiplicador)
    cost_tank_unit = lng_tank_cost  # Usamos el input del usuario
    cost_civil_per_tank = 100000 * fuel_infra_mult   # Obra civil escala con el factor
    cost_vaporizers = 50000 * fuel_infra_mult        # Equipos menores escalan
    cost_piping_controls = 500000 * fuel_infra_mult  # BOP escala
    
    # 2. Cálculo
    n_tanks = lng_metrics.get("num_tanks", 1)
    storage_capex = n_tanks * (cost_tank_unit + cost_civil_per_tank)
    
    peak_flow = lng_metrics.get("peak_flow_mmbtu_hr", 100)
    vaporizers_needed = math.ceil(peak_flow / 50) + 1 
    regas_capex = vaporizers_needed * cost_vaporizers
    
    # Total LNG CAPEX
    lng_infra_cost = storage_capex + regas_capex + cost_piping_controls
    lng_capex_m = lng_infra_cost / 1e6

# Costo de Gasoducto (Si aplica)
if not is_lng_primary:
    # Costo base por metro lineal ($50/m para tubo pequeño, escalado por diámetro)
    base_pipe_cost = 50 * (rec_pipe_dia if rec_pipe_dia > 0 else 4) 
    
    # Aplicamos el multiplicador de infraestructura (zanjas, permisos, terreno difícil)
    adjusted_pipe_cost = base_pipe_cost * fuel_infra_mult
    
    pipeline_capex_m = (adjusted_pipe_cost * dist_gas_main_m) / 1e6

# CAPEX breakdown ACTUALIZADO
cost_items = [
    {"Item": "Generation Units", "Index": 1.00, "Cost (M USD)": gen_cost_total},
    {"Item": "Installation & BOP", "Index": idx_install, "Cost (M USD)": gen_cost_total * idx_install},
    {"Item": "Tri-Gen Plant", "Index": idx_chp, "Cost (M USD)": gen_cost_total * idx_chp},
    {"Item": "BESS System", "Index": 0.0, "Cost (M USD)": bess_capex_m},
    {"Item": "Fuel Infra (LNG/Pipe)", "Index": 0.0, "Cost (M USD)": lng_capex_m + pipeline_capex_m},
    {"Item": "Emissions Control", "Index": 0.0, "Cost (M USD)": at_capex_total / 1e6},
]
df_capex = pd.DataFrame(cost_items)
initial_capex_sum = df_capex["Cost (M USD)"].sum()

# Tax benefits
itc_benefit_m = (initial_capex_sum * 0.30) if (enable_itc and include_chp) else 0
depreciation_benefit_m = calculate_macrs_depreciation(initial_capex_sum * 1e6, project_years) / 1e6 if enable_depreciation else 0

# Repowering (BESS replacements)
repowering_pv_m = 0.0
if use_bess:
    for year in range(1, project_years + 1):
        year_cost = 0.0
        if year % bess_life_batt == 0 and year < project_years:
            year_cost += (bess_energy_total * 1000 * bess_cost_kwh)
        if year % bess_life_inv == 0 and year < project_years:
            year_cost += (bess_power_total * 1000 * bess_cost_kw)
        if year_cost > 0:
            repowering_pv_m += (year_cost / 1e6) / ((1 + wacc) ** year)

# Annualized costs
crf = (wacc * (1 + wacc)**project_years) / ((1 + wacc)**project_years - 1)
repowering_annualized = repowering_pv_m * 1e6 * crf

# ENHANCED O&M CALCULATION
effective_hours = 8760 * capacity_factor
mwh_year = p_total_avg * effective_hours

# O&M Fixed ($/kW-year)
om_fixed_kw_yr = 15.0
om_fixed_annual = (installed_cap * 1000) * om_fixed_kw_yr

# O&M Variable ($/MWh)
om_variable_mwh = 3.5
om_variable_annual = mwh_year * om_variable_mwh

# O&M Labor
om_labor_per_unit = 120000
om_labor_annual = n_total * om_labor_per_unit

# Major overhaul (60k hours)
overhaul_interval_years = 60000 / (8760 * capacity_factor)
overhaul_cost_per_mw = 150000
overhaul_pv = 0
for year in np.arange(overhaul_interval_years, project_years, overhaul_interval_years):
    year_int = int(year)
    cost = installed_cap * overhaul_cost_per_mw
    overhaul_pv += cost / ((1 + wacc) ** year_int)
overhaul_annualized = overhaul_pv * crf

om_cost_year = om_fixed_annual + om_variable_annual + om_labor_annual + bess_om_annual + overhaul_annualized

# Fuel costs with degradation
fuel_cost_year = total_fuel_input_mmbtu_hr * total_gas_price * effective_hours

# Carbon costs
carbon_cost_year = co2_ton_yr * carbon_price_per_ton

# Total annual cost
capex_annualized = (initial_capex_sum * 1e6) * crf
total_annual_cost = fuel_cost_year + om_cost_year + capex_annualized + repowering_annualized + carbon_cost_year

# Tax benefits (reduce annual cost)
ptc_annual = (mwh_year * 1000 * 0.013) if enable_ptc else 0
itc_annualized = (itc_benefit_m * 1e6) * crf
depreciation_annualized = (depreciation_benefit_m * 1e6) * crf

total_annual_cost_after_tax = total_annual_cost - ptc_annual - itc_annualized - depreciation_annualized

# LCOE
lcoe = total_annual_cost_after_tax / (mwh_year * 1000)

# NPV
annual_grid_cost = mwh_year * 1000 * benchmark_price
annual_savings = annual_grid_cost - (fuel_cost_year + om_cost_year + carbon_cost_year)

if wacc > 0:
    pv_savings = annual_savings * ((1 - (1 + wacc)**-project_years) / wacc)
else:
    pv_savings = annual_savings * project_years

# Add tax benefits to NPV
total_tax_benefits = (itc_benefit_m + depreciation_benefit_m) * 1e6 + (ptc_annual * project_years)

npv = pv_savings + total_tax_benefits - (initial_capex_sum * 1e6) - (repowering_pv_m * 1e6)

if annual_savings > 0:
    payback_years = (initial_capex_sum * 1e6) / annual_savings
    roi_simple = (annual_savings / (initial_capex_sum * 1e6)) * 100
    payback_str = f"{payback_years:.1f} Years"
else:
    payback_str = "N/A"
    roi_simple = 0

# Gas price sensitivity
gas_prices_x = np.linspace(0, total_gas_price * 2, 20)
lcoe_y = []
for gp in gas_prices_x:
    sim_fuel = total_fuel_input_mmbtu_hr * gp * effective_hours
    sim_total = sim_fuel + om_cost_year + capex_annualized + repowering_annualized + carbon_cost_year
    sim_total_after_tax = sim_total - ptc_annual - itc_annualized - depreciation_annualized
    sim_lcoe = sim_total_after_tax / (mwh_year * 1000)
    lcoe_y.append(sim_lcoe)

breakeven_gas_price = 0.0
for gp in gas_prices_x:
    sim_fuel = total_fuel_input_mmbtu_hr * gp * effective_hours
    sim_total = sim_fuel + om_cost_year + capex_annualized + repowering_annualized + carbon_cost_year
    sim_total_after_tax = sim_total - ptc_annual - itc_annualized - depreciation_annualized
    sim_lcoe = sim_total_after_tax / (mwh_year * 1000)
    if sim_lcoe <= benchmark_price:
        breakeven_gas_price = gp
        break

# ==============================================================================
# 8. OUTPUTS - ENHANCED TABBED INTERFACE
# ==============================================================================

t1, t2, t3, t4, t5 = st.tabs([
    "📊 System Design", 
    "⚡ Performance & Stability", 
    "🏗️ Footprint & Optimization",
    "❄️ Cooling & Tri-Gen", 
    "💰 Economics & ROI"
])

with t1:
    st.subheader("System Architecture")
    
    # KPIs
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Generator", selected_gen)
    c2.metric("Fleet", f"{n_running}+{n_reserve}")
    c3.metric("Installed", f"{installed_cap:.1f} MW")
    
    # Availability with status indicator
    if target_met:
        c4.metric("Availability", f"{prob_gen*100:.3f}%", delta="✅ Target Met", delta_color="normal")
    else:
        c4.metric("Availability", f"{prob_gen*100:.3f}%", delta="⚠️ Below Target", delta_color="inverse")
    
    c5.metric("PUE", f"{pue_actual:.2f}")
    c6.metric("Density", f"{gen_data['power_density_mw_per_m2']*1000:.0f} kW/m²")
    
    # ===========================================================================
    # NEW: SPINNING RESERVE IMPACT VISUALIZATION
    # ===========================================================================
    st.markdown("### 🔄 Spinning Reserve & Load Distribution")
    
    col_sr1, col_sr2, col_sr3, col_sr4 = st.columns(4)
    
    col_sr1.metric(
        "Spinning Reserve Required",
        f"{selected_config.get('spinning_reserve_mw', spinning_reserve_mw):.1f} MW",
        f"{spinning_res_pct:.0f}% of avg load"  # <--- CORREGIDO (Usa la variable de política)
    )
    
    col_sr2.metric(
        "From Generators",
        f"{selected_config.get('spinning_from_gens', spinning_from_gens):.1f} MW",
        "Headroom in running units"
    )
    
    col_sr3.metric(
        "From BESS",
        f"{selected_config.get('spinning_from_bess', spinning_from_bess):.1f} MW" if use_bess else "N/A",
        "Instant response" if use_bess else ""
    )
    
    col_sr4.metric(
        "Load/Unit",
        f"{load_per_unit_pct:.1f}%",
        f"Headroom: {100 - load_per_unit_pct:.1f}%"
    )
    
    # Explanation of spinning reserve impact
    if use_bess:
        units_diff = best_config_a['n_running'] - selected_config['n_running'] if best_config_a else 0
        load_diff = selected_config['load_pct'] - best_config_a['load_pct'] if best_config_a else 0
        
        if units_diff > 0 or load_diff > 0:
            st.success(
                f"✅ **BESS Spinning Reserve Benefit:**\n\n"
                f"- Without BESS: {best_config_a['n_running']} running units at {best_config_a['load_pct']:.1f}% load\n"
                f"- With BESS: {selected_config['n_running']} running units at {selected_config['load_pct']:.1f}% load\n"
                f"- **Result:** {units_diff} fewer units, {load_diff:.1f}% higher load = better efficiency!"
            )
    else:
        st.info(
            f"ℹ️ **Spinning Reserve Strategy (No BESS):**\n\n"
            f"- Running {n_running} units at {load_per_unit_pct:.1f}% load each\n"
            f"- Headroom for spinning reserve: {selected_config.get('headroom_mw', n_running * unit_site_cap - p_total_avg):.1f} MW\n"
            f"- All spinning reserve must come from generator headroom"
        )
    
    # Visual: Load Distribution Bar Chart
    fig_load_dist = go.Figure()
    
    # For each running unit
    load_per_unit_mw = p_total_avg / n_running
    headroom_per_unit_mw = unit_site_cap - load_per_unit_mw
    
    fig_load_dist.add_trace(go.Bar(
        name='Actual Load',
        x=[f'Gen {i+1}' for i in range(min(n_running, 10))],  # Show max 10 units
        y=[load_per_unit_mw] * min(n_running, 10),
        marker_color='#1f77b4',
        text=[f'{load_per_unit_pct:.1f}%'] * min(n_running, 10),
        textposition='inside'
    ))
    
    fig_load_dist.add_trace(go.Bar(
        name='Spinning Reserve (Headroom)',
        x=[f'Gen {i+1}' for i in range(min(n_running, 10))],
        y=[headroom_per_unit_mw] * min(n_running, 10),
        marker_color='#90EE90',
        text=[f'{100-load_per_unit_pct:.1f}%'] * min(n_running, 10),
        textposition='inside'
    ))
    
    if use_bess and bess_power_total > 0:
        fig_load_dist.add_trace(go.Bar(
            name='BESS Reserve',
            x=['BESS'],
            y=[bess_power_total],
            marker_color='#FFD700',
            text=[f'{bess_power_total:.1f} MW'],
            textposition='inside'
        ))
    
    fig_load_dist.update_layout(
        title=f"Load Distribution Across {n_running} Running Units (showing up to 10)",
        barmode='stack',
        yaxis_title='Power (MW)',
        height=400
    )
    
    fig_load_dist.add_hline(
        y=unit_site_cap,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Unit Capacity: {unit_site_cap:.1f} MW"
    )
    
    st.plotly_chart(fig_load_dist, use_container_width=True)
    
    # Load Profile Visualization
    st.markdown("### 📈 Annual Load Profile & Duration Curve")
    
    hours = np.arange(0, 8760)
    daily_wave = 1.0 + 0.15 * np.sin(2 * np.pi * hours / 24 - np.pi/2)
    load_curve = p_total_avg * daily_wave * np.random.uniform(0.95, 1.05, len(hours))
    load_curve = np.clip(load_curve, 0, p_total_peak)
    load_sorted = np.sort(load_curve)[::-1]
    
    fig_ldc = go.Figure()
    fig_ldc.add_trace(go.Scatter(
        x=hours, y=load_sorted, fill='tozeroy',
        name='DC Load', line=dict(color='#667eea', width=2)
    ))
    fig_ldc.add_hline(
        y=installed_cap, line_dash="dash", line_color="red",
        annotation_text=f"Total Installed: {installed_cap:.1f} MW",
        annotation_position="top right"
    )
    
    # Show genset online capacity line
    genset_online_capacity = n_running * unit_site_cap
    fig_ldc.add_hline(
        y=genset_online_capacity, line_dash="dashdot", line_color="green",
        annotation_text=f"Online Capacity: {genset_online_capacity:.1f} MW",
        annotation_position="bottom right"
    )
    
    fig_ldc.add_hline(
        y=p_total_avg, line_dash="dot", line_color="orange",
        annotation_text=f"Average: {p_total_avg:.1f} MW"
    )
    
    # Show spinning reserve zone
    fig_ldc.add_hrect(
        y0=p_total_avg, y1=p_total_avg + spinning_reserve_mw,
        fillcolor="lightgreen", opacity=0.3,
        annotation_text=f"Spinning Reserve: {spinning_reserve_mw:.1f} MW",
        annotation_position="top left"
    )
    
    # Show BESS peak shaving zone if applicable
    if use_bess and p_total_peak > genset_online_capacity:
        fig_ldc.add_hrect(
            y0=genset_online_capacity, y1=p_total_peak,
            fillcolor="yellow", opacity=0.2,
            annotation_text=f"BESS Peak Shaving Zone ({bess_power_total:.1f} MW)",
            annotation_position="top left"
        )
    
    fig_ldc.update_layout(
        title="Load Duration Curve with Spinning Reserve Visualization",
        xaxis_title="Hours per Year (Sorted)",
        yaxis_title="Load (MW)",
        height=400
    )
    st.plotly_chart(fig_ldc, use_container_width=True)
    
    # Fleet details
    st.markdown("### 🔧 Fleet Configuration Details")
    
    col_f1, col_f2 = st.columns(2)
    
    with col_f1:
        st.markdown("**Generator Specifications:**")
        st.write(f"- Model: {selected_gen}")
        st.write(f"- Type: {gen_data['type']}")
        st.write(f"- ISO Rating: {unit_iso_cap:.2f} MW")
        st.write(f"- Site Rating: {unit_site_cap:.2f} MW")
        st.write(f"- Efficiency (ISO): {gen_data['electrical_efficiency']*100:.1f}%")
        st.write(f"- Ramp Rate: {gen_data['ramp_rate_mw_s']:.1f} MW/s")
        st.write(f"- MTBF: {gen_data['mtbf_hours']:,} hours")
    
    with col_f2:
        st.markdown("**Operating Parameters:**")
        st.write(f"- Strategy: {load_strategy}")
        st.write(f"- Units Running: {units_running} of {n_total}")
        st.write(f"- **Load per Unit: {load_per_unit_pct:.1f}%** ← Spinning Reserve Impact")
        st.write(f"- Fleet Efficiency: {fleet_efficiency*100:.1f}%")
        st.write(f"- Capacity Factor: {capacity_factor*100:.0f}%")
        st.write(f"- Hours/Year: {effective_hours:.0f}")
        st.write(f"- Annual Energy: {mwh_year:,.0f} MWh")

    # --- VISUALIZACIÓN LOGÍSTICA LNG ---
    if has_lng_storage and lng_metrics:
        st.markdown("---")
        st.subheader("🚛 LNG Logistics & Supply Chain")
        
        c_lng1, c_lng2, c_lng3, c_lng4 = st.columns(4)
        
        c_lng1.metric("Daily Consumption", f"{lng_metrics['gal_per_day']:,.0f} Gal")
        c_lng2.metric("Storage Required", f"{lng_metrics['storage_gal']:,.0f} Gal", f"{lng_days} Days Autonomy")
        c_lng3.metric("Infrastructure", f"{lng_metrics['num_tanks']} Tanks", f"60k Gal each")
        
        # Semáforo de tráfico
        trucks = lng_metrics['trucks_day']
        if trucks < 1:
             truck_delta = "Low Traffic"
             truck_color = "normal"
        elif trucks < 5:
             truck_delta = "Medium Traffic"
             truck_color = "off"
        else:
             truck_delta = "High Traffic!"
             truck_color = "inverse"
             
        if is_lng_primary:
            c_lng4.metric("Trucks per Day", f"{trucks:.1f}", truck_delta, delta_color=truck_color)
            st.info(f"ℹ️ **Logistics Plan:** You need approximately **{int(math.ceil(lng_metrics['trucks_week']))} deliveries per week**. Ensure road access allows heavy trucks.")
        else:
            c_lng4.metric("Trucks per Week", "< 1", "Backup Mode")
            st.success("✅ **Backup Mode:** LNG usage is minimal. Fuel is stored for emergencies.")
    
with t2:
    st.subheader("Electrical Performance & Stability")
    
    # ========================================================================
    # RELIABILITY TRADE-OFFS TABLE
    # ========================================================================
    if len(reliability_configs) > 1:
        st.markdown("### ⚖️ Reliability Configuration Comparison")
        
        # Build comparison table with spinning reserve info
        comparison_data = []
        for config in reliability_configs:
            # Calculate CAPEX (CORREGIDO: Incluye Instalación y BOP)
            # Costo Unitario Total = Equipo + Instalación
            unit_total_cost_kw = gen_data['est_cost_kw'] + gen_data['est_install_kw']
            
            genset_capex = config['n_total'] * unit_site_cap * unit_total_cost_kw / 1000
            
            # BESS Capex: Potencia ($250/kW) + Energía ($400/kWh)
            bess_cost_kw = 250.0
            bess_cost_kwh = 400.0
            bess_capex = (config['bess_mw'] * 1000 * bess_cost_kw + config['bess_mwh'] * 1000 * bess_cost_kwh) / 1e6
            
            total_capex = genset_capex + bess_capex
            
            # Calculate O&M/year
            genset_om = config['n_total'] * 120
            bess_om = config['bess_mwh'] * 10
            total_om = (genset_om + bess_om) / 1000
            
            running_units = config['n_running']
            
            # Get load_pct from config (now properly calculated)
            load_per_unit = config.get('load_pct', (p_total_avg / (running_units * unit_site_cap)) * 100)
            
            if 'efficiency' in config:
                config_efficiency = config['efficiency']
            else:
                config_efficiency = get_part_load_efficiency(
                    gen_data["electrical_efficiency"],
                    load_per_unit,
                    gen_data["type"]
                )
            
            annual_energy_gwh = (p_total_avg * 8760 * capacity_factor) / 1000
            annual_fuel_mmbtu = (annual_energy_gwh * 1000 * 3.412) / config_efficiency
            
            comparison_data.append({
                'Configuration': config['name'],
                'Fleet': f"{config['n_running']}+{config['n_reserve']}",
                'Total Units': config['n_total'],
                'BESS (MW/MWh)': f"{config['bess_mw']:.0f}/{config['bess_mwh']:.0f}" if config['bess_mw'] > 0 else "None",
                'Load/Unit (%)': f"{load_per_unit:.1f}%",
                'Spinning from BESS': f"{config.get('spinning_from_bess', 0):.1f} MW" if config.get('spinning_from_bess', 0) > 0 else "-",
                'Fleet Eff (%)': f"{config_efficiency*100:.1f}%",
                'Availability': f"{config['availability']*100:.3f}%",
                'CAPEX (M$)': f"${total_capex:.1f}M",
            })
        
        df_comparison = pd.DataFrame(comparison_data)
        
        # Highlight selected configuration
        selected_name = selected_config['name']
        
        st.dataframe(
            df_comparison.style.apply(
                lambda row: ['background-color: #d4edda' if row['Configuration'] == selected_name else '' for _ in row],
                axis=1
            ),
            use_container_width=True,
            hide_index=True
        )
        
        # Show selected config details
        col_sel1, col_sel2, col_sel3 = st.columns(3)
        
        with col_sel1:
            st.metric(
                "✅ Selected Configuration",
                selected_name,
                delta=f"{selected_config['n_total']} total units"
            )
        
        with col_sel2:
            load_improvement = selected_config['load_pct'] - best_config_a['load_pct'] if best_config_a else 0
            if load_improvement > 0:
                st.metric(
                    "Load/Unit Improvement",
                    f"+{load_improvement:.1f}%",
                    delta=f"Higher efficiency from BESS"
                )
            else:
                st.metric("Load/Unit", f"{selected_config['load_pct']:.1f}%")
        
        with col_sel3:
            st.metric(
                "Availability Achieved",
                f"{selected_config['availability']*100:.4f}%",
                delta="✅ Target Met" if target_met else "⚠️ Below Target",
                delta_color="normal" if target_met else "inverse"
            )
        
        st.markdown("---")
    
    col_e1, col_e2, col_e3, col_e4 = st.columns(4)
    col_e1.metric("Voltage", f"{rec_voltage_kv} kV")
    col_e2.metric("Frequency", f"{freq_hz} Hz")
    col_e3.metric("X\"d", f"{gen_data['reactance_xd_2']:.3f} pu")
    col_e4.metric("Ramp Rate", f"{gen_data['ramp_rate_mw_s']:.1f} MW/s")
    
    # Net Efficiency & Heat Rate
    st.markdown("### ⚙️ Net Efficiency & Heat Rate")
    
    # 1. Recuperamos tus inputs dinámicos
    # Si el usuario no editó nada, usa 2.0% por defecto, si no, usa lo que puso en el Expander.
    aux_power_pct = gen_data.get("aux_pct", 2.0) 
    
    # Recuperamos el valor del Slider de pérdidas que creaste en el Paso 1
    current_dist_loss = dist_loss_pct 
    
    # 2. Cálculos de Eficiencia Real
    gross_efficiency = fleet_efficiency 
    
    # Eficiencia Neta = Eficiencia Motor * (1 - Consumo Propio) * (1 - Pérdidas Cables/Trafos)
    # Ejemplo: 45% * (1 - 0.03) * (1 - 0.015) = ~43% entregado al servidor
    net_efficiency = gross_efficiency * (1 - aux_power_pct/100) * (1 - current_dist_loss/100)
    
    # 3. Conversión a Heat Rate (Combustible)
    if net_efficiency > 0:
        heat_rate_lhv_btu = 3412 / net_efficiency
    else:
        heat_rate_lhv_btu = 0
        
    heat_rate_hhv_btu = heat_rate_lhv_btu * 1.11 # Valor típico HHV/LHV para gas natural
    heat_rate_lhv_mj = heat_rate_lhv_btu * 0.001055
    heat_rate_hhv_mj = heat_rate_hhv_btu * 0.001055
    
    col_eff1, col_eff2, col_eff3, col_eff4 = st.columns(4)
    
    with col_eff1:
        st.metric("Gross Efficiency", f"{gross_efficiency*100:.2f}%")
        st.caption("At generator terminals")
    
    with col_eff2:
        st.metric("Net Efficiency", f"{net_efficiency*100:.2f}%")
        st.caption("After auxiliaries")
    
    with col_eff3:
        if is_imperial:
            st.metric("Heat Rate (HHV)", f"{heat_rate_hhv_btu:.0f}")
            st.caption("BTU/kWh")
        else:
            st.metric("Heat Rate (HHV)", f"{heat_rate_hhv_mj:.2f}")
            st.caption("MJ/kWh")
    
    with col_eff4:
        if is_imperial:
            st.metric("Heat Rate (LHV)", f"{heat_rate_lhv_btu:.0f}")
            st.caption("BTU/kWh")
        else:
            st.metric("Heat Rate (LHV)", f"{heat_rate_lhv_mj:.2f}")
            st.caption("MJ/kWh")
    
    st.markdown("---")
    
    # Transient Stability
    st.markdown("### 🎯 Transient Stability Analysis")
    
    if stability_ok:
        st.success(f"✅ **Voltage Sag OK:** {voltage_sag:.2f}% (Limit: 10%)")
    else:
        st.error(f"❌ **Voltage Sag Exceeds:** {voltage_sag:.2f}% > 10%")
        st.warning("**Mitigation:** Add generators, increase BESS, or use lower X\"d units")
    
    # Step Load & BESS
    st.markdown("### 🔋 Step Load Capability & BESS Analysis")
    
    col_s1, col_s2, col_s3 = st.columns(3)
    # Usamos load_step_pct porque aquí hablamos del GOLPE FÍSICO
    col_s1.metric("Max Load Step", f"{load_step_pct:.0f}%")
    col_s2.metric("Gen Capability", f"{gen_data['step_load_pct']:.0f}%")
    
    # Usamos load_step_pct porque comparamos capacidad FÍSICA
    step_capable = gen_data["step_load_pct"] >= load_step_pct
    if step_capable:
        col_s3.success("✅ COMPLIANT")
    elif use_bess:
        col_s3.warning("⚠️ BESS REQUIRED")
    else:
        col_s3.error("❌ NOT COMPLIANT")
    
    if use_bess:
        st.info(f"🔋 **BESS Capacity:** {bess_power_total:.1f} MW / {bess_energy_total:.1f} MWh")
        
        # BESS Breakdown
        # FIX: Add Reliability Backup to the chart data
        bess_breakdown_data = pd.DataFrame({
            'Component': ['Step Support', 'Peak Shaving', 'Ramp Support', 'Freq Reg', 'Spinning Reserve', 'Black Start', 'Reliability Backup'],
            'Power (MW)': [
                bess_breakdown.get('step_support', 0),
                bess_breakdown.get('peak_shaving', 0),
                bess_breakdown.get('ramp_support', 0),
                bess_breakdown.get('freq_reg', 0),
                bess_breakdown.get('spinning_reserve', 0),
                bess_breakdown.get('black_start', 0),
                bess_breakdown.get('reliability_backup', 0) # <--- ¡NUEVA LÍNEA!
            ]
        })
        
        # Filter out zero values
        bess_breakdown_data = bess_breakdown_data[bess_breakdown_data['Power (MW)'] > 0.01]
        
        col_bess1, col_bess2 = st.columns([2, 1])
        
        with col_bess1:
            fig_bess = px.bar(bess_breakdown_data, x='Component', y='Power (MW)',
                             title="BESS Sizing Breakdown (Dominant Component Sets Capacity)", 
                             color='Component',
                             text='Power (MW)')
            fig_bess.update_traces(texttemplate='%{text:.1f} MW', textposition='outside')
            fig_bess.update_layout(showlegend=False, height=350)
            st.plotly_chart(fig_bess, use_container_width=True)
        
        with col_bess2:
            st.markdown("**BESS Functions:**")
            if bess_breakdown.get('spinning_reserve', 0) > 0:
                st.write(f"✅ **Spinning Reserve: {bess_breakdown['spinning_reserve']:.1f} MW**")
            if bess_breakdown.get('peak_shaving', 0) > 0:
                st.write(f"✅ Peak Shaving: {bess_breakdown['peak_shaving']:.1f} MW")
            if bess_breakdown.get('step_support', 0) > 0:
                st.write(f"✅ Step Load: {bess_breakdown['step_support']:.1f} MW")
            if bess_breakdown.get('ramp_support', 0) > 0:
                st.write(f"✅ Ramp Rate: {bess_breakdown['ramp_support']:.1f} MW")
            if bess_breakdown.get('freq_reg', 0) > 0:
                st.write(f"✅ Freq Reg: {bess_breakdown['freq_reg']:.1f} MW")
            if bess_breakdown.get('black_start', 0) > 0:
                st.write(f"✅ Black Start: {bess_breakdown['black_start']:.1f} MW")
            
            st.caption(f"ℹ️ Final capacity = MAX of all requirements")

    
    # Reliability over time (Weibull)
    st.markdown("### 📊 Reliability Projection (Aging Model)")
    
    if not target_met:
        st.error(f"🛑 **Availability Target Not Achieved**")
        col_gap1, col_gap2, col_gap3 = st.columns(3)
        col_gap1.metric("Target", f"{avail_req:.3f}%")
        col_gap2.metric("Achieved", f"{prob_gen*100:.3f}%", f"-{(avail_req - prob_gen*100):.3f}%")
        col_gap3.metric("Configuration", f"N+{n_reserve} (Maximum)")
    else:
        st.success(f"✅ **Availability Target Met:** {prob_gen*100:.3f}% ≥ {avail_req:.3f}% with N+{n_reserve}")
    
    years_range = list(range(1, project_years + 1))
    
    fig_rel = go.Figure()
    fig_rel.add_trace(go.Scatter(
        x=years_range, y=[a*100 for a in availability_curve],
        mode='lines', name='Availability',
        line=dict(color='#007bff', width=3)
    ))
    fig_rel.add_hline(
        y=avail_req, line_dash="dash", line_color="red",
        annotation_text=f"Target: {avail_req:.2f}%"
    )
    fig_rel.update_layout(
        title="System Availability Over Time (with 0.5%/year degradation)",
        xaxis_title="Project Year",
        yaxis_title="Availability (%)",
        height=400
    )
    st.plotly_chart(fig_rel, use_container_width=True)
    
    # Emissions
    st.markdown("### 🌍 Environmental Performance")
    
    col_em1, col_em2, col_em3, col_em4 = st.columns(4)
    col_em1.metric("NOx", f"{nox_lb_hr:.2f} lb/hr")
    col_em2.metric("CO", f"{co_lb_hr:.2f} lb/hr")
    col_em3.metric("CO₂/Year", f"{co2_ton_yr:,.0f} tons")
    col_em4.metric("Carbon Cost", f"${carbon_cost_year/1e6:.2f}M/yr")
    
    if at_capex_total > 0:
        st.warning(f"⚠️ **Emissions Control:** SCR + Catalyst (${at_capex_total/1e6:.2f}M)")

with t3:
    st.subheader("Footprint Analysis & Optimization")
    
    col_fp1, col_fp2, col_fp3 = st.columns(3)
    col_fp1.metric("Total Footprint", f"{disp_area:.2f} {disp_area_unit}")
    col_fp2.metric("Power Density", f"{gen_data['power_density_mw_per_m2']:.3f} MW/m²")
    
    if enable_footprint_limit:
        col_fp3.metric("Utilization", f"{area_utilization_pct:.1f}%")
    else:
        col_fp3.metric("Status", "No Limit Set")
    
    # Footprint breakdown
    st.markdown("### 📐 Footprint Breakdown")
    
    footprint_data = pd.DataFrame({
        "Component": ["Generators", "BESS", "Fuel Storage", "Cooling/CHP", "Substation", "Contingency (20%)"],
        "Area (m²)": [area_gen, area_bess, storage_area_m2, area_chp, area_sub, total_area_m2 * 0.2]
    })
    
    col_pie1, col_pie2 = st.columns([2, 1])
    
    with col_pie1:
        fig_pie = px.pie(footprint_data, values='Area (m²)', names='Component',
                        title=f"Total: {disp_area:.2f} {disp_area_unit}")
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col_pie2:
        st.dataframe(footprint_data, use_container_width=True)
    
    if is_area_exceeded and footprint_recommendations:
        st.error(f"🛑 **Footprint Exceeded:** {total_area_m2:,.0f} m² > {max_area_m2:,.0f} m² ({(total_area_m2/max_area_m2-1)*100:.0f}% over)")
        
        st.markdown("### 💡 Optimization Recommendations")
        
        for i, rec in enumerate(footprint_recommendations, 1):
            with st.expander(f"Option {i}: {rec['action']} (Saves {rec['savings_pct']:.1f}%)"):
                st.write(f"**Type:** {rec['type']}")
                st.write(f"**New Footprint:** {rec['new_area']:,.0f} m² ({rec['new_area']/10000:.2f} Ha)")
                st.write(f"**Savings:** {rec['savings_pct']:.1f}%")
                st.write(f"**Trade-off:** {rec['trade_off']}")
    
    elif enable_footprint_limit:
        st.success(f"✅ **Footprint OK:** {area_utilization_pct:.1f}% of available area")

with t4:
    st.subheader("Cooling & Tri-Gen")
    
    # --- NUEVO: VALIDACIÓN DE INGENIERÍA (REALITY CHECK) ---
    if include_chp and cooling_method == "Air-Cooled":
        st.warning(
            "⚠️ **Engineering Conflict Detected:**\n\n"
            "You have selected **Air-Cooled** but enabled **Tri-Generation (CHP)**.\n"
            "- CHP uses waste heat for Absorption Chillers, which typically require water towers (Water-Cooled).\n"
            "- **Reality Check:** Ensure site has water availability for the CHP loop, even if IT cooling is air-based."
        )
    # -------------------------------------------------------
    
    col_c1, col_c2, col_c3, col_c4 = st.columns(4)
    
    if include_chp:
        col_c1.metric("Heat Recovered", f"{total_heat_rec_mw:.1f} MWt")
        col_c2.metric("Cooling Generated", f"{total_cooling_mw_chp:.1f} MWc")
        col_c3.metric("Coverage", f"{cooling_coverage_pct:.1f}%")
        col_c4.metric("PUE Improvement", f"{pue - pue_actual:.2f}")
        
        st.progress(min(1.0, cooling_coverage_pct/100))
        
        st.info(f"💡 **Tri-Gen Benefit:** PUE reduced from {pue:.2f} to {pue_actual:.2f}")
    else:
        col_c1.metric("Cooling Method", cooling_method)
        col_c2.metric("Cooling Load", f"{total_cooling_mw:.1f} MWc")
        col_c3.metric("PUE", f"{pue_actual:.2f}")
        col_c4.metric("WUE", f"{wue:.1f}")
    
    st.metric(f"Water Consumption (WUE {wue:.1f})", f"{disp_water:,.0f} gal/day" if is_imperial else f"{disp_water:,.0f} m³/day")
    
    if wue > 1.5:
        st.warning("⚠️ **High Water Use:** Consider dry cooling or recycling")

with t5:
    st.subheader("Financial Analysis & Economics")
    
    # Tax benefits summary
    if itc_benefit_m > 0 or depreciation_benefit_m > 0 or ptc_annual > 0:
        st.markdown("### 💸 Tax Benefits & Incentives")
        
        col_tax1, col_tax2, col_tax3, col_tax4 = st.columns(4)
        
        if itc_benefit_m > 0:
            col_tax1.metric("ITC (30%)", f"${itc_benefit_m:.2f}M")
        if depreciation_benefit_m > 0:
            col_tax2.metric("MACRS Depreciation", f"${depreciation_benefit_m:.2f}M")
        if ptc_annual > 0:
            col_tax3.metric("PTC (Annual)", f"${ptc_annual/1e6:.2f}M")
        
        total_tax_benefit_m = itc_benefit_m + depreciation_benefit_m + (ptc_annual * project_years / 1e6)
        col_tax4.metric("Total Tax Benefit", f"${total_tax_benefit_m:.2f}M")
    
   # ==========================================================================
    # LCOE GAP ANALYSIS & RECOMMENDER SYSTEM
    # ==========================================================================
    if enable_lcoe_target and target_lcoe > 0:
        lcoe_gap = lcoe - target_lcoe
        
        if lcoe_gap > 0:
            st.error(f"⚠️ **Target Missed:** LCOE ${lcoe:.4f}/kWh > Target ${target_lcoe:.4f}/kWh (Gap: +${lcoe_gap:.4f})")
            
            st.markdown("### 💡 Strategies to Reduce LCOE")
            rec_count = 0
            
            # 1. ANALISIS DE EFICIENCIA (High Efficiency vs Fast Response)
            # Si la eficiencia es < 44% y el gas es caro (> $3.5), sugerir cambio de motor
            if gen_data['electrical_efficiency'] < 0.44 and total_gas_price > 3.5:
                # Estimar ahorro por pasar a G3520K (45.3%)
                fuel_saving_pct = 1 - (gen_data['electrical_efficiency'] / 0.453)
                potential_saving = (fuel_cost_year * abs(fuel_saving_pct)) / (mwh_year * 1000)
                
                if potential_saving > 0.002:
                    st.info(f"🔹 **Upgrade to High Efficiency:** Switching to **G3520K** or **G20CM34** could save ~${potential_saving:.4f}/kWh in fuel.")
                    rec_count += 1

            # 2. CREDITOS FISCALES (ITC / CHP)
            # Si no tiene CHP activo, está perdiendo el 30% de ITC y eficiencia térmica
            if not include_chp:
                # Estimar impacto del ITC (30% CAPEX Credit)
                potential_itc = (initial_capex_sum * 0.30 * 1e6) * crf / (mwh_year * 1000)
                if potential_itc > 0.003:
                    st.info(f"🔹 **Enable Tri-Generation (CHP):** Adding heat recovery unlocks **30% ITC Tax Credit**, reducing LCOE by ~${potential_itc:.4f}/kWh.")
                    rec_count += 1

            # 3. FINANCIAMIENTO (WACC)
            # El WACC al 8% es estándar, pero si es >7%, reducirlo impacta mucho
            if wacc > 0.07:
                # Estimar impacto de bajar WACC a 5%
                low_wacc = 0.05
                crf_low = (low_wacc * (1 + low_wacc)**project_years) / ((1 + low_wacc)**project_years - 1)
                capex_saving = ((initial_capex_sum * 1e6) * (crf - crf_low)) / (mwh_year * 1000)
                st.info(f"🔹 **Refinance Project:** Lowering WACC from {wacc*100:.1f}% to 5.0% would save ${capex_saving:.4f}/kWh.")
                rec_count += 1
            
            # 4. PRECIO DEL GAS
            if total_gas_price > 4.5:
                st.info(f"🔹 **Gas Contract:** Your gas price (${total_gas_price:.2f}) is high. Negotiating down to $3.50 would save ${(total_gas_price - 3.50) * 10:.4f}/kWh approx.")
                rec_count += 1
                
            # 5. UTILIZACIÓN (Capacity Factor)
            if capacity_factor < 0.80 and dc_type != "Edge Computing":
                 st.info(f"🔹 **Increase Utilization:** Increasing Capacity Factor spreads fixed CAPEX over more kWh. Current: {capacity_factor*100:.0f}%.")
                 rec_count += 1

            if rec_count == 0:
                st.warning("⚠️ LCOE is optimized for these conditions. Consider extending Project Life or seeking Grants.")

        else:
            st.success(f"🎉 **Target Met:** LCOE ${lcoe:.4f}/kWh < Target ${target_lcoe:.4f}/kWh")
    
    # CAPEX Editor
    st.markdown("### 💰 CAPEX Breakdown")
    st.info(f"**Regional Multiplier:** {region} ({regional_mult:.2f}x)")
    
    edited_capex = st.data_editor(
        df_capex,
        column_config={
            "Index": st.column_config.NumberColumn("Multiplier", min_value=0.0, max_value=5.0, step=0.01),
            "Cost (M USD)": st.column_config.NumberColumn("Cost", format="$%.2fM", disabled=True)
        },
        use_container_width=True
    )
    
    total_capex_dynamic = edited_capex["Cost (M USD)"].sum()
    
    # Recalculate financials
    capex_annualized_dyn = (total_capex_dynamic * 1e6) * crf
    total_annual_cost_dyn = fuel_cost_year + om_cost_year + capex_annualized_dyn + repowering_annualized + carbon_cost_year
    total_annual_cost_dyn_after_tax = total_annual_cost_dyn - ptc_annual - itc_annualized - depreciation_annualized
    lcoe_dyn = total_annual_cost_dyn_after_tax / (mwh_year * 1000)
    
    npv_dyn = pv_savings + total_tax_benefits - (total_capex_dynamic * 1e6) - (repowering_pv_m * 1e6)
    
    if annual_savings > 0:
        payback_dyn = (total_capex_dynamic * 1e6) / annual_savings
        roi_dyn = (annual_savings / (total_capex_dynamic * 1e6)) * 100
        payback_str_dyn = f"{payback_dyn:.1f} Years"
    else:
        payback_str_dyn = "N/A"
        roi_dyn = 0
    
    # Financial KPIs
    st.markdown("### 📊 Key Financial Metrics")
    
    c_f1, c_f2, c_f3, c_f4, c_f5 = st.columns(5)
    c_f1.metric("CAPEX", f"${total_capex_dynamic:.2f}M")
    c_f2.metric("LCOE", f"${lcoe_dyn:.4f}/kWh")
    c_f3.metric("Annual Savings", f"${annual_savings/1e6:.2f}M")
    c_f4.metric("NPV (20yr)", f"${npv_dyn/1e6:.2f}M")
    c_f5.metric("Payback", payback_str_dyn)
    
    # O&M Breakdown
    st.markdown("### 🔧 Annual O&M Breakdown")
    
    om_data = pd.DataFrame({
        'Component': ['Fixed ($/kW-yr)', 'Variable ($/MWh)', 'Labor', 'Major Overhaul', 'BESS O&M'],
        'Annual Cost ($M)': [
            om_fixed_annual/1e6,
            om_variable_annual/1e6,
            om_labor_annual/1e6,
            overhaul_annualized/1e6,
            bess_om_annual/1e6
        ]
    })
    
    fig_om = px.bar(om_data, x='Component', y='Annual Cost ($M)',
                   title=f"Total O&M: ${om_cost_year/1e6:.2f}M/year",
                   color='Component')
    fig_om.update_layout(showlegend=False, height=350)
    st.plotly_chart(fig_om, use_container_width=True)
    
    # LCOE Breakdown
    st.markdown("### 💵 LCOE Component Breakdown")
    
    cost_data = pd.DataFrame({
        "Component": ["Fuel", "O&M", "CAPEX", "Repowering", "Carbon", "Tax Benefits"],
        "$/kWh": [
            fuel_cost_year/(mwh_year*1000),
            om_cost_year/(mwh_year*1000),
            capex_annualized_dyn/(mwh_year*1000),
            repowering_annualized/(mwh_year*1000),
            carbon_cost_year/(mwh_year*1000),
            -(ptc_annual + itc_annualized + depreciation_annualized)/(mwh_year*1000)
        ]
    })
    
    fig_lcoe = px.bar(cost_data, x="Component", y="$/kWh",
                     title="LCOE Breakdown", text_auto='.4f',
                     color="Component")
    fig_lcoe.update_layout(showlegend=False, height=400)
    st.plotly_chart(fig_lcoe, use_container_width=True)
    
    # Gas Price Sensitivity
    st.markdown("### 📈 Gas Price Sensitivity Analysis")
    
    if breakeven_gas_price > 0:
        st.success(f"🎯 **Breakeven Gas Price:** ${breakeven_gas_price:.2f}/MMBtu")
    else:
        st.error("⚠️ **No Breakeven:** Prime power more expensive even with free gas")
    
    fig_sens = go.Figure()
    fig_sens.add_trace(go.Scatter(
        x=gas_prices_x, y=lcoe_y, mode='lines',
        name='LCOE (Prime)', line=dict(color='#667eea', width=3)
    ))
    fig_sens.add_hline(
        y=benchmark_price, line_dash="dash", line_color="red",
        annotation_text=f"Benchmark: ${benchmark_price:.3f}/kWh"
    )
    if breakeven_gas_price > 0:
        fig_sens.add_vline(
            x=breakeven_gas_price, line_dash="dot", line_color="green",
            annotation_text=f"Breakeven: ${breakeven_gas_price:.2f}"
        )
    fig_sens.update_layout(
        title="LCOE vs Gas Price",
        xaxis_title="Total Gas Price ($/MMBtu)",
        yaxis_title="LCOE ($/kWh)",
        height=450
    )
    st.plotly_chart(fig_sens, use_container_width=True)

    # =========================================================================
    # 📄 COMPREHENSIVE PDF REPORT GENERATION (ReportLab)
    # =========================================================================
    st.markdown("---")
    st.markdown("## 📄 Export Comprehensive Report")
    
    col_exp1, col_exp2 = st.columns([3, 1])
    
    with col_exp1:
        project_name_input = st.text_input(
            "Project Name", 
            value=f"{dc_type} - {p_it:.0f}MW Data Center",
            help="Enter a project name for the report"
        )
    
    def generate_comprehensive_pdf():
        """Generate a complete PDF report with all sizing results"""
        
        buffer = BytesIO()
        doc = SimpleDocTemplate(
            buffer, 
            pagesize=letter,
            rightMargin=0.6*inch, 
            leftMargin=0.6*inch,
            topMargin=0.6*inch, 
            bottomMargin=0.6*inch
        )
        
        # Styles
        styles = getSampleStyleSheet()
        
        # Custom styles
        styles.add(ParagraphStyle(
            name='ReportTitle',
            parent=styles['Title'],
            fontSize=22,
            textColor=HexColor('#1a1a2e'),
            spaceAfter=15,
            alignment=TA_CENTER
        ))
        
        styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=styles['Heading1'],
            fontSize=13,
            textColor=HexColor('#FFFFFF'),
            spaceBefore=15,
            spaceAfter=8,
            backColor=HexColor('#1a1a2e'),
            borderPadding=6
        ))
        
        styles.add(ParagraphStyle(
            name='SubSection',
            parent=styles['Heading2'],
            fontSize=11,
            textColor=HexColor('#333333'),
            spaceBefore=12,
            spaceAfter=6
        ))
        
        styles.add(ParagraphStyle(
            name='CustomBody',
            parent=styles['Normal'],
            fontSize=9,
            textColor=HexColor('#333333'),
            spaceAfter=6,
            alignment=TA_JUSTIFY
        ))
        
        story = []
        
        # =====================================================================
        # COVER PAGE
        # =====================================================================
        story.append(Spacer(1, 1*inch))
        story.append(Paragraph("⚡ CAT SIZE SOLUTION", styles['ReportTitle']))
        story.append(Paragraph("Comprehensive Power System Sizing Report", styles['Heading2']))
        story.append(Spacer(1, 0.4*inch))
        
        # Project info
        cover_data = [
            ['Project:', project_name_input],
            ['Data Center Type:', dc_type],
            ['Report Date:', datetime.now().strftime("%B %d, %Y")],
            ['Region:', region],
            ['Generated By:', 'CAT Size Solution v3.0']
        ]
        
        cover_table = Table(cover_data, colWidths=[1.8*inch, 4.2*inch])
        cover_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('ALIGN', (0, 0), (0, -1), 'RIGHT'),
            ('ALIGN', (1, 0), (1, -1), 'LEFT'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))
        story.append(cover_table)
        story.append(Spacer(1, 0.5*inch))
        
        # Executive Summary
        summary_text = f"""
        <b>Executive Summary:</b><br/><br/>
        This report presents the power system sizing for a <b>{p_it:.0f} MW</b> IT load data center 
        with <b>PUE {pue:.2f}</b>. The recommended solution: <b>{n_total} x {selected_gen}</b> 
        in <b>N+{n_reserve}</b> configuration achieving <b>{prob_gen*100:.3f}%</b> availability.
        {'<br/><br/>Includes <b>' + f'{bess_power_total:.1f} MW / {bess_energy_total:.1f} MWh</b> BESS for transient support.' if use_bess else ''}
        <br/><br/>
        <b>Key Metrics:</b> LCOE ${lcoe:.4f}/kWh | CAPEX ${initial_capex_sum:.1f}M | Payback {payback_str}
        """
        story.append(Paragraph(summary_text, styles['CustomBody']))
        story.append(PageBreak())
        
        # =====================================================================
        # SECTION 1: LOAD REQUIREMENTS
        # =====================================================================
        story.append(Paragraph("1. LOAD REQUIREMENTS", styles['SectionHeader']))
        
        load_data = [
            ['Parameter', 'Value', 'Unit'],
            ['Critical IT Load', f'{p_it:.1f}', 'MW'],
            ['Power Usage Effectiveness (PUE)', f'{pue:.2f}', '-'],
            ['Total DC Load (Design)', f'{p_total_dc:.1f}', 'MW'],
            ['Average Operating Load', f'{p_total_avg:.1f}', 'MW'],
            ['Peak Load', f'{p_total_peak:.1f}', 'MW'],
            ['Capacity Factor', f'{capacity_factor*100:.1f}', '%'],
            ['Required Availability', f'{avail_req:.4f}', '%'],
            ['Step Load Requirement', f'{load_step_pct:.0f}', '%'],
            ['Spinning Reserve Policy', f'{spinning_res_pct:.0f}', '%'],
        ]
        
        load_table = Table(load_data, colWidths=[2.8*inch, 2*inch, 1.5*inch])
        load_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), HexColor('#1a1a2e')),
            ('TEXTCOLOR', (0, 0), (-1, 0), white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#cccccc')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [white, HexColor('#f8f8f8')]),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
            ('TOPPADDING', (0, 0), (-1, -1), 5),
        ]))
        story.append(load_table)
        story.append(Spacer(1, 0.25*inch))
        
        # =====================================================================
        # SECTION 2: GENERATOR SELECTION
        # =====================================================================
        story.append(Paragraph("2. GENERATOR SELECTION", styles['SectionHeader']))
        
        gen_data_table = [
            ['Specification', 'Value'],
            ['Model', selected_gen],
            ['Type', gen_data.get('type', 'N/A')],
            ['ISO Rating', f"{gen_data.get('iso_rating_mw', 0):.2f} MW"],
            ['Site Rating (Derated)', f"{unit_site_cap:.2f} MW"],
            ['Derate Factor', f"{derate_factor_calc*100:.1f}%"],
            ['Electrical Efficiency', f"{gen_data.get('electrical_efficiency', 0)*100:.1f}%"],
            ['Heat Rate (LHV)', f"{gen_data.get('heat_rate_lhv', 0):,.0f} BTU/kWh"],
            ['Step Load Capability', f"{gen_data.get('step_load_pct', 0):.0f}%"],
            ['Ramp Rate', f"{gen_data.get('ramp_rate_mw_s', 0):.1f} MW/s"],
            ['MTBF', f"{gen_data.get('mtbf_hours', 0):,} hours"],
        ]
        
        gen_table = Table(gen_data_table, colWidths=[3*inch, 3.3*inch])
        gen_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), HexColor('#1a1a2e')),
            ('TEXTCOLOR', (0, 0), (-1, 0), white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('ALIGN', (1, 1), (1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#cccccc')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [white, HexColor('#f8f8f8')]),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ]))
        story.append(gen_table)
        story.append(Spacer(1, 0.25*inch))
        
        # =====================================================================
        # SECTION 3: FLEET CONFIGURATION
        # =====================================================================
        story.append(Paragraph("3. FLEET CONFIGURATION", styles['SectionHeader']))
        
        fleet_data = [
            ['Parameter', 'Value'],
            ['Running Units (N)', f'{n_running}'],
            ['Reserve Units (+X)', f'{n_reserve}'],
            ['Total Fleet', f'{n_total}'],
            ['Installed Capacity', f'{installed_cap:.1f} MW'],
            ['Load per Unit', f'{load_per_unit_pct:.1f}%'],
            ['Fleet Efficiency', f'{fleet_efficiency*100:.2f}%'],
            ['Configuration', f'N+{n_reserve}'],
            ['Achieved Availability', f'{prob_gen*100:.4f}%'],
            ['Target Met', 'YES ✓' if target_met else 'NO ✗'],
        ]
        
        fleet_table = Table(fleet_data, colWidths=[3*inch, 3.3*inch])
        fleet_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), HexColor('#1a1a2e')),
            ('TEXTCOLOR', (0, 0), (-1, 0), white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('ALIGN', (1, 1), (1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#cccccc')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [white, HexColor('#f8f8f8')]),
        ]))
        story.append(fleet_table)
        story.append(Spacer(1, 0.25*inch))
        
        # =====================================================================
        # SECTION 4: SPINNING RESERVE ANALYSIS
        # =====================================================================
        story.append(Paragraph("4. SPINNING RESERVE ANALYSIS", styles['SectionHeader']))
        
        spin_data = [
            ['Parameter', 'Value', 'Notes'],
            ['Spinning Reserve Required', f"{selected_config.get('spinning_reserve_mw', 0):.1f} MW", f'{spinning_res_pct:.0f}% of avg load'],
            ['From Generators (Headroom)', f"{selected_config.get('spinning_from_gens', 0):.1f} MW", 'Running units headroom'],
            ['From BESS', f"{selected_config.get('spinning_from_bess', 0):.1f} MW" if use_bess else 'N/A', 'Instant response'],
            ['Available Headroom', f"{selected_config.get('headroom_mw', 0):.1f} MW", 'Total spare capacity'],
        ]
        
        spin_table = Table(spin_data, colWidths=[2.3*inch, 1.8*inch, 2.2*inch])
        spin_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), HexColor('#1a1a2e')),
            ('TEXTCOLOR', (0, 0), (-1, 0), white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#cccccc')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [white, HexColor('#f8f8f8')]),
        ]))
        story.append(spin_table)
        
        story.append(PageBreak())
        
        # =====================================================================
        # SECTION 5: CONFIGURATION COMPARISON
        # =====================================================================
        if len(reliability_configs) > 1:
            story.append(Paragraph("5. CONFIGURATION COMPARISON", styles['SectionHeader']))
            
            config_header = ['Configuration', 'Fleet', 'BESS (MW/MWh)', 'Load/Unit', 'Efficiency', 'Availability']
            config_rows = [config_header]
            
            for cfg in reliability_configs:
                bess_str = f"{cfg.get('bess_mw', 0):.0f}/{cfg.get('bess_mwh', 0):.0f}" if cfg.get('bess_mw', 0) > 0 else "None"
                config_rows.append([
                    cfg.get('name', 'N/A'),
                    f"{cfg.get('n_running', 0)}+{cfg.get('n_reserve', 0)}",
                    bess_str,
                    f"{cfg.get('load_pct', 0):.1f}%",
                    f"{cfg.get('efficiency', 0)*100:.1f}%",
                    f"{cfg.get('availability', 0)*100:.3f}%"
                ])
            
            config_table = Table(config_rows, colWidths=[1.4*inch, 0.7*inch, 1*inch, 0.9*inch, 0.9*inch, 1*inch])
            config_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), HexColor('#1a1a2e')),
                ('TEXTCOLOR', (0, 0), (-1, 0), white),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 8),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#cccccc')),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [white, HexColor('#f8f8f8')]),
            ]))
            story.append(config_table)
            story.append(Spacer(1, 0.25*inch))
        
        # =====================================================================
        # SECTION 6: BESS SYSTEM
        # =====================================================================
        if use_bess:
            story.append(Paragraph("6. BESS SYSTEM", styles['SectionHeader']))
            
            bess_data = [
                ['Parameter', 'Value'],
                ['Strategy', bess_strategy],
                ['Power Capacity', f'{bess_power_total:.1f} MW'],
                ['Energy Capacity', f'{bess_energy_total:.1f} MWh'],
                ['Duration', f'{bess_energy_total/bess_power_total:.1f} hours' if bess_power_total > 0 else 'N/A'],
                ['Step Load Support', f"{bess_breakdown.get('step_support', 0):.1f} MW"],
                ['Peak Shaving', f"{bess_breakdown.get('peak_shaving', 0):.1f} MW"],
                ['Spinning Reserve', f"{bess_breakdown.get('spinning_reserve', 0):.1f} MW"],
            ]
            
            bess_table = Table(bess_data, colWidths=[3*inch, 3.3*inch])
            bess_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), HexColor('#1a1a2e')),
                ('TEXTCOLOR', (0, 0), (-1, 0), white),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('ALIGN', (1, 1), (1, -1), 'CENTER'),
                ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#cccccc')),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [white, HexColor('#f8f8f8')]),
            ]))
            story.append(bess_table)
            story.append(Spacer(1, 0.25*inch))
        
        # =====================================================================
        # SECTION 7: ELECTRICAL PERFORMANCE
        # =====================================================================
        story.append(Paragraph("7. ELECTRICAL PERFORMANCE", styles['SectionHeader'] if use_bess else styles['SectionHeader']))
        
        elec_data = [
            ['Parameter', 'Value'],
            ['Connection Voltage', f'{rec_voltage_kv} kV'],
            ['System Frequency', f'{freq_hz} Hz'],
            ['Transient Stability', 'PASS ✓' if stability_ok else 'FAIL ✗'],
            ['Voltage Sag', f'{voltage_sag:.2f}% (Limit: 10%)'],
            ['Net Efficiency', f'{net_efficiency*100:.2f}%'],
            ['Heat Rate (HHV)', f'{heat_rate_hhv_mj:.2f} MJ/kWh'],
        ]
        
        elec_table = Table(elec_data, colWidths=[3*inch, 3.3*inch])
        elec_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), HexColor('#1a1a2e')),
            ('TEXTCOLOR', (0, 0), (-1, 0), white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('ALIGN', (1, 1), (1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#cccccc')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [white, HexColor('#f8f8f8')]),
        ]))
        story.append(elec_table)
        story.append(Spacer(1, 0.25*inch))
        
        # =====================================================================
        # SECTION 8: FOOTPRINT
        # =====================================================================
        story.append(Paragraph("8. FOOTPRINT & INFRASTRUCTURE", styles['SectionHeader']))
        
        foot_data = [
            ['Component', 'Area (m²)', 'Area (Acres)'],
            ['Generators', f'{area_gen:,.0f}', f'{area_gen * 0.000247105:.2f}'],
            ['BESS', f'{area_bess:,.0f}' if use_bess else 'N/A', f'{area_bess * 0.000247105:.2f}' if use_bess else 'N/A'],
            ['Total Site', f'{total_area_m2:,.0f}', f'{total_area_m2 * 0.000247105:.2f}'],
        ]
        
        foot_table = Table(foot_data, colWidths=[2.3*inch, 2*inch, 2*inch])
        foot_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), HexColor('#1a1a2e')),
            ('TEXTCOLOR', (0, 0), (-1, 0), white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#cccccc')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [white, HexColor('#f8f8f8')]),
        ]))
        story.append(foot_table)
        
        story.append(PageBreak())
        
        # =====================================================================
        # SECTION 9: ENVIRONMENTAL
        # =====================================================================
        story.append(Paragraph("9. ENVIRONMENTAL PERFORMANCE", styles['SectionHeader']))
        
        env_data = [
            ['Parameter', 'Value', 'Unit'],
            ['NOx Emissions', f'{nox_lb_hr:.2f}', 'lb/hr'],
            ['CO Emissions', f'{co_lb_hr:.2f}', 'lb/hr'],
            ['CO2 Emissions (Annual)', f'{co2_ton_yr:,.0f}', 'tons/yr'],
            ['Carbon Cost', f'${carbon_cost_year/1e6:.2f}M', '/year'],
            ['Cooling Method', 'Tri-Gen (CHP)' if include_chp else cooling_method, '-'],
            ['Actual PUE', f'{pue_actual:.2f}', '-'],
        ]
        
        env_table = Table(env_data, colWidths=[2.5*inch, 2*inch, 1.8*inch])
        env_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), HexColor('#1a1a2e')),
            ('TEXTCOLOR', (0, 0), (-1, 0), white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#cccccc')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [white, HexColor('#f8f8f8')]),
        ]))
        story.append(env_table)
        story.append(Spacer(1, 0.25*inch))
        
        # =====================================================================
        # SECTION 10: FINANCIAL ANALYSIS
        # =====================================================================
        story.append(Paragraph("10. FINANCIAL ANALYSIS", styles['SectionHeader']))
        
        fin_data = [
            ['Metric', 'Value'],
            ['Total CAPEX', f'${initial_capex_sum:.2f}M'],
            ['LCOE', f'${lcoe:.4f}/kWh'],
            ['Annual Fuel Cost', f'${fuel_cost_year/1e6:.2f}M'],
            ['Annual O&M Cost', f'${om_cost_year/1e6:.2f}M'],
            ['Annual Savings vs Grid', f'${annual_savings/1e6:.2f}M'],
            ['NPV (Project Life)', f'${npv/1e6:.2f}M'],
            ['Simple Payback', payback_str],
            ['WACC', f'{wacc*100:.1f}%'],
            ['Project Life', f'{project_years} years'],
            ['Gas Price', f'${total_gas_price:.2f}/MMBtu'],
            ['Benchmark Electricity', f'${benchmark_price:.3f}/kWh'],
        ]
        
        fin_table = Table(fin_data, colWidths=[3*inch, 3.3*inch])
        fin_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), HexColor('#1a1a2e')),
            ('TEXTCOLOR', (0, 0), (-1, 0), white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('ALIGN', (1, 1), (1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#cccccc')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [white, HexColor('#f8f8f8')]),
        ]))
        story.append(fin_table)
        story.append(Spacer(1, 0.25*inch))
        
        if breakeven_gas_price > 0:
            story.append(Paragraph(f"<b>Breakeven Gas Price:</b> ${breakeven_gas_price:.2f}/MMBtu", styles['CustomBody']))
        
        story.append(PageBreak())
        
        # =====================================================================
        # SECTION 11: CAPEX BREAKDOWN
        # =====================================================================
        story.append(Paragraph("11. CAPEX BREAKDOWN", styles['SectionHeader']))
        
        capex_rows = [['Item', 'Cost (M USD)']]
        for _, row in df_capex.iterrows():
            capex_rows.append([row['Item'], f"${row['Cost (M USD)']:.2f}M"])
        capex_rows.append(['TOTAL', f"${initial_capex_sum:.2f}M"])
        
        capex_table = Table(capex_rows, colWidths=[4*inch, 2.3*inch])
        capex_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), HexColor('#1a1a2e')),
            ('TEXTCOLOR', (0, 0), (-1, 0), white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold'),
            ('BACKGROUND', (0, -1), (-1, -1), HexColor('#e8e8e8')),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('ALIGN', (1, 0), (1, -1), 'RIGHT'),
            ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#cccccc')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -2), [white, HexColor('#f8f8f8')]),
        ]))
        story.append(capex_table)
        story.append(Spacer(1, 0.4*inch))
        
        # =====================================================================
        # DISCLAIMER
        # =====================================================================
        story.append(Paragraph("NOTES & DISCLAIMER", styles['SectionHeader']))
        
        disclaimer_text = """
        <b>Important Notes:</b><br/>
        1. This analysis is preliminary and for planning purposes only.<br/>
        2. Actual equipment selection requires detailed engineering studies.<br/>
        3. Site conditions may affect performance.<br/>
        4. Financial projections based on current market assumptions.<br/><br/>
        
        <b>Disclaimer:</b> This report is provided for informational purposes only. 
        Final system design should be validated by qualified engineers. 
        Caterpillar Inc. makes no warranties regarding accuracy or completeness.
        """
        story.append(Paragraph(disclaimer_text, styles['CustomBody']))
        
        # Build PDF
        doc.build(story)
        buffer.seek(0)
        return buffer.getvalue()
    
    with col_exp2:
        st.write("")
        if st.button("📥 Generate PDF Report", type="primary", use_container_width=True):
            with st.spinner("Generating comprehensive PDF..."):
                try:
                    pdf_data = generate_comprehensive_pdf()
                    st.success("✅ PDF generated!")
                    
                    st.download_button(
                        label="📄 Download Report",
                        data=pdf_data,
                        file_name=f"CAT_SizeSolution_{project_name_input.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    st.caption("Report includes: Load Requirements, Generator Selection, Fleet Configuration, Spinning Reserve Analysis, BESS System, Electrical Performance, Footprint, Environmental, and Financial Analysis.")

# --- FOOTER ---
st.markdown("---")
col_foot1, col_foot2, col_foot3 = st.columns(3)
col_foot1.caption("CAT Size Solution v3.0")
col_foot2.caption("Next-Gen Data Center Power Solutions")
col_foot3.caption("Caterpillar Electric Power | 2026")

















