#!/usr/bin/env python3
import os
import sys
import numpy as np

try:
    import pinocchio as pin
except ImportError:
    print("Error: 'pinocchio' library not found. Install it via: pip3 install pinocchio", file=sys.stderr)
    sys.exit(1)


# ===========================================================================
# CHASSIS PHYSICAL PARAMETERS  (from URDF inertial_link + base_link geometry)
#
# The inertial_link (fixed child of base_link) carries all chassis inertia:
#   mass        = 2.1557 kg
#   Ixx = 0.24  kg·m²   (roll)
#   Iyy = 0.96  kg·m²   (pitch)
#   Izz = 0.96  kg·m²   (yaw  ← relevant for planar motion)
#
# When JointModelPlanar() is used as the root joint, Pinocchio lumps the
# inertia of ALL fixed-child links (base_link + inertial_link + sensor links)
# into the first movable body.  The planar joint adds 3 DOF:
#   v[0] = ẋ   (forward velocity,  m/s)
#   v[1] = ẏ   (lateral velocity,  m/s)
#   v[2] = ω_z (yaw rate,          rad/s)
# ===========================================================================

# Mass declared in inertial_link (chassis)
CHASSIS_MASS_KG = 2.1557
CHASSIS_IZZ     = 0.96      # kg·m²  — yaw inertia (planar motion)

JOINT_DESCRIPTIONS = {
    # ---- Planar root joint (added by JointModelPlanar) -------------------
    "root_joint": {
        "dof_type"    : "Planar (SE2 root)",
        "axes"        : "X, Y, Z-rot",
        "parent_link" : "world",
        "child_link"  : "base_footprint",
        "mass_kg"     : CHASSIS_MASS_KG,
        "inertia_izz" : CHASSIS_IZZ,
        "note"        : (
            "Virtual root joint added by JointModelPlanar(). "
            "Represents the 2-D rigid-body motion of the whole chassis: "
            "translation (x, y) + yaw (θ_z). "
            "Inertia comes from inertial_link + base_link + all fixed sensor links."
        ),
    },
    # ---- Wheel / steering joints -----------------------------------------
    "rear_left_wheel": {
        "dof_type"    : "Continuous (wheel)",
        "axes"        : "Y",
        "parent_link" : "base_link",
        "child_link"  : "rear_left_wheel_link",
        "position_xyz": "(-0.1,  0.07, -0.101) m from base_link",
        "mass_kg"     : 0.50,
        "inertia_izz" : 0.01055,
        "note"        : "Rear left drive wheel. Free spin, velocity-controlled.",
    },
    "left_steering_hinge_wheel": {
        "dof_type"    : "Revolute (steering)",
        "axes"        : "Z",
        "parent_link" : "base_link",
        "child_link"  : "left_steering_hinge",
        "position_xyz": "( 0.1,  0.07, -0.101) m from base_link",
        "mass_kg"     : 0.25,
        "inertia_izz" : 0.00525,
        "note"        : "Left Ackermann steering hinge. Limits ±30° (±0.5236 rad), max effort 5 Nm.",
    },
    "front_left_wheel": {
        "dof_type"    : "Continuous (wheel)",
        "axes"        : "Y",
        "parent_link" : "left_steering_hinge",
        "child_link"  : "front_left_wheel_link",
        "position_xyz": "(0.0, 0.0, 0.0) m from left_steering_hinge",
        "mass_kg"     : 0.25,
        "inertia_izz" : 0.00525,
        "note"        : "Steering front left wheel. Inherits orientation from the parent hinge.",
    },
    "front_right_wheel": {
        "dof_type"    : "Continuous (wheel)",
        "axes"        : "-Y",
        "parent_link" : "right_steering_hinge",
        "child_link"  : "front_right_wheel_link",
        "position_xyz": "(0.0, 0.0, 0.0) m from right_steering_hinge",
        "mass_kg"     : 0.25,
        "inertia_izz" : 0.00525,
        "note"        : "Steering front right wheel. Axis inverted w.r.t. left side.",
    },
    "rear_right_wheel": {
        "dof_type"    : "Continuous (wheel)",
        "axes"        : "-Y",
        "parent_link" : "base_link",
        "child_link"  : "rear_right_wheel_link",
        "position_xyz": "(-0.1, -0.07, -0.101) m from base_link",
        "mass_kg"     : 0.50,
        "inertia_izz" : 0.01055,
        "note"        : "Rear right drive wheel. RPY=(π,0,0) → -Y axis for sign consistency.",
    },
    "right_steering_hinge_wheel": {
        "dof_type"    : "Revolute (steering)",
        "axes"        : "-Z",
        "parent_link" : "base_link",
        "child_link"  : "right_steering_hinge",
        "position_xyz": "( 0.1, -0.07, -0.101) m from base_link",
        "mass_kg"     : 0.25,
        "inertia_izz" : 0.00525,
        "note"        : "Right Ackermann steering hinge. RPY=(π,0,0) → -Z axis for consistency.",
    },
}

# ---------------------------------------------------------------------------
# Detailed description of each element of the PLANAR root joint.
# The planar joint uses a non-standard q layout: [x, y, cos(θ), sin(θ)]  (nq=4)
# and a standard v layout: [ẋ, ẏ, ω_z]  (nv=3).
# ---------------------------------------------------------------------------
PLANAR_Q_DESCRIPTIONS = [
    "x      [m]      — chassis position along world X axis",
    "y      [m]      — chassis position along world Y axis",
    "cos(θ) [adim]   — cosine of chassis yaw angle (θ=0 → pointing +X)",
    "sin(θ) [adim]   — sine   of chassis yaw angle",
]
PLANAR_V_DESCRIPTIONS = [
    "ẋ      [m/s]    — chassis linear velocity along world X",
    "ẏ      [m/s]    — chassis linear velocity along world Y",
    "ω_z    [rad/s]  — chassis yaw rate (positive = counter-clockwise)",
]


# ===========================================================================
# HELPERS
# ===========================================================================

def find_urdf():
    """Recursively searches for limo_ackerman.urdf in standard workspace directories."""
    search_roots = [
        os.path.expanduser('~/limo_project'),
        '/catkin_ws',
        '/ros2_ws',
    ]
    target = 'limo_ackerman.urdf'
    for root in search_roots:
        if not os.path.exists(root):
            continue
        for dirpath, _, filenames in os.walk(root):
            if target in filenames:
                found = os.path.join(dirpath, target)
                print(f"[INFO] URDF found at: {found}")
                return found
    raise FileNotFoundError(f"File '{target}' not found in: {search_roots}")


def build_joint_map(model):
    """
    Returns a sorted list of dicts with joint metadata for all active joints.
    Handles the planar root joint (nq=4, nv=3) explicitly.
    """
    joint_map = []
    v_idx = 0
    for jnt_id in range(1, model.njoints):
        jnt  = model.joints[jnt_id]
        nv_j = jnt.nv
        if nv_j == 0:
            continue
        name = model.names[jnt_id]
        joint_map.append({
            "name"  : name,
            "idx_v" : v_idx,
            "nv"    : nv_j,
            "idx_q" : jnt.idx_q,
            "nq"    : jnt.nq,
        })
        v_idx += nv_j
    return joint_map


def format_q_explained(q, joint_map):
    """
    Formats vector q with per-element explanations.
    Handles: planar root (nq=4), continuous wheels (nq=2), revolute steering (nq=1).
    """
    lines = ["  Index  Value     Joint / Meaning"]
    lines.append("  -----  --------  ------------------------------------------------")

    for jnt in joint_map:
        name  = jnt["name"]
        idx_q = jnt["idx_q"]
        nq_j  = jnt["nq"]
        desc  = JOINT_DESCRIPTIONS.get(name, {})
        dof_t = desc.get("dof_type", "n/a")

        if nq_j == 4:
            # Planar joint: [x, y, cos(θ), sin(θ)]
            for k, meaning in enumerate(PLANAR_Q_DESCRIPTIONS):
                lines.append(
                    f"  q[{idx_q+k:2d}]  {q[idx_q+k]:+.4f}  {name} → {meaning}"
                )
        elif nq_j == 2:
            # Continuous (wheel): [cos θ, sin θ]
            lines.append(f"  q[{idx_q:2d}]  {q[idx_q]:+.4f}  {name} → cos(θ)  [{dof_t}]")
            lines.append(f"  q[{idx_q+1:2d}]  {q[idx_q+1]:+.4f}  {name} → sin(θ)  (θ=0 in neutral)")
        elif nq_j == 1:
            # Revolute (steering): scalar θ
            lines.append(f"  q[{idx_q:2d}]  {q[idx_q]:+.4f}  {name} → θ [rad]  [{dof_t}]")
        else:
            for k in range(nq_j):
                lines.append(f"  q[{idx_q+k:2d}]  {q[idx_q+k]:+.4f}  {name} (component {k})")

    return "\n".join(lines)


def format_v_explained(v, joint_map):
    """
    Formats vector v with per-element explanations.
    Handles: planar root (nv=3), wheels/steering (nv=1 each).
    """
    lines = ["  Index  Value     Joint / Meaning"]
    lines.append("  -----  --------  ------------------------------------------------")

    for jnt in joint_map:
        name  = jnt["name"]
        idx_v = jnt["idx_v"]
        nv_j  = jnt["nv"]
        desc  = JOINT_DESCRIPTIONS.get(name, {})
        dof_t = desc.get("dof_type", "n/a")
        axes  = desc.get("axes", "n/a")

        if nv_j == 3:
            # Planar root joint
            for k, meaning in enumerate(PLANAR_V_DESCRIPTIONS):
                lines.append(
                    f"  v[{idx_v+k:2d}]  {v[idx_v+k]:+.4f}  {name} → {meaning}"
                )
        else:
            for k in range(nv_j):
                lines.append(
                    f"  v[{idx_v+k:2d}]  {v[idx_v+k]:+.4f}  "
                    f"{name} → dθ/dt [rad/s]  axis {axes}  [{dof_t}]"
                )

    return "\n".join(lines)


def format_matrix_explained(M, joint_map, matrix_name, unit):
    """
    Prints the full matrix and a diagonal breakdown with physical meaning.
    For the planar root block (3×3), prints the full sub-block.
    """
    np.set_printoptions(precision=4, suppress=True, linewidth=140)
    lines = [f"\n  {matrix_name} [{unit}]  —  {M.shape[0]}×{M.shape[1]} matrix:"]
    lines.append(np.array2string(M, prefix="  "))

    lines.append("\n  Diagonal breakdown (physical interpretation):")
    lines.append(
        "  DOF  Value        Joint                           Physical meaning"
    )
    lines.append(
        "  ---  -----------  ------------------------------  ----------------------------------------"
    )

    for jnt in joint_map:
        name  = jnt["name"]
        idx_v = jnt["idx_v"]
        nv_j  = jnt["nv"]
        desc  = JOINT_DESCRIPTIONS.get(name, {})

        if nv_j == 3:
            # Planar root: show the full 3×3 block on the diagonal
            block_labels = [
                f"M_xx = {M[idx_v,   idx_v  ]:.4f}  chassis total mass (x direction) [kg]",
                f"M_yy = {M[idx_v+1, idx_v+1]:.4f}  chassis total mass (y direction) [kg]",
                f"M_zz = {M[idx_v+2, idx_v+2]:.4f}  chassis yaw inertia (Izz lumped) [kg·m²]",
            ]
            for k, label in enumerate(block_labels):
                lines.append(f"  {idx_v+k:3d}  {M[idx_v+k,idx_v+k]:+.6f}   {name:<30s}  {label}")
            # Off-diagonal of the 3×3 block
            lines.append(
                f"        M_xy = {M[idx_v,idx_v+1]:.4f}  M_xz = {M[idx_v,idx_v+2]:.4f}  "
                f"M_yz = {M[idx_v+1,idx_v+2]:.4f}  (off-diagonal planar block)"
            )
        else:
            izz  = desc.get("inertia_izz", None)
            note = f"Izz from URDF ≈ {izz:.5f}" if izz is not None else ""
            lines.append(
                f"  {idx_v:3d}  {M[idx_v,idx_v]:+.6f}   {name:<30s}  {note}"
            )

    return "\n".join(lines)


def format_gravity_explained(g, joint_map):
    """Formats the gravity vector with per-DOF explanation."""
    lines = ["  Index  Value [N or N·m]  Joint / Meaning"]
    lines.append("  -----  ---------------  ------------------------------------------------")

    for jnt in joint_map:
        name  = jnt["name"]
        idx_v = jnt["idx_v"]
        nv_j  = jnt["nv"]

        if nv_j == 3:
            meaning = [
                "force  along X  [N]   — gravity has no X component on a flat robot",
                "force  along Y  [N]   — gravity has no Y component on a flat robot",
                "torque around Z [N·m] — gravity has no yaw torque (CoM on Z axis)",
            ]
            for k, m in enumerate(meaning):
                lines.append(f"  g[{idx_v+k:2d}]  {g[idx_v+k]:+.6f}        {name} → {m}")
        else:
            lines.append(
                f"  g[{idx_v:2d}]  {g[idx_v]:+.6f}        {name} → "
                f"gravitational torque [N·m]  (0 because CoM is on rotation axis)"
            )

    return "\n".join(lines)


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    # 1. Locate URDF
    try:
        urdf_file = find_urdf()
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)

    # 2. Build Pinocchio model with PLANAR root joint
    #    This adds 3 DOF (x, y, θ_z) as the first joint, representing the
    #    rigid-body planar motion of the whole chassis in the world frame.
    try:
        model = pin.buildModelFromUrdf(urdf_file, pin.JointModelPlanar())
        data  = model.createData()
        print(f"[INFO] Model loaded (planar base) — "
              f"joints: {model.njoints}, nq: {model.nq}, nv: {model.nv}")
        print(f"[INFO] Expected: nq = {3+1 + 4*2 + 2} = 14 "
              f"(planar[x,y,cos,sin] + 4 wheels×[cos,sin] + 2 hinges×[θ])")
        print(f"[INFO]           nv = {3 + 4 + 2} = 9  "
              f"(planar[ẋ,ẏ,ω] + 4 wheels + 2 hinges)")
    except Exception as e:
        print(f"[ERROR] Unable to build Pinocchio model: {e}", file=sys.stderr)
        sys.exit(1)

    # 3. Map joints → indices
    joint_map = build_joint_map(model)

    # 4. State: neutral configuration, zero velocity
    q = pin.neutral(model)
    v = np.zeros(model.nv)

    # 5. Compute dynamic matrices
    M = pin.crba(model, data, q)
    M = (M + M.T) - np.diag(np.diag(M))        # enforce symmetry
    C = pin.computeCoriolisMatrix(model, data, q, v)
    g = pin.computeGeneralizedGravity(model, data, q)

    # 5b. Extract friction parameters from model
    # Pinocchio reads <dynamics damping="..." friction="..."/> from the URDF.
    # The planar root joint (index 1) has no friction declared → 0.
    # damping[i]  = viscous coefficient  b_i   [N·m·s/rad]
    # friction[i] = Coulomb coefficient  f_i   [N·m]
    #
    # NOTE: model.damping and model.friction are indexed by joint id (1..njoints-1),
    # NOT by DOF index. We need to map them onto the nv-dimensional v vector.
    damping_vec  = np.zeros(model.nv)   # b  vector, length nv
    friction_vec = np.zeros(model.nv)   # f  vector, length nv
    for jnt_id in range(1, model.njoints):
        jnt  = model.joints[jnt_id]
        nv_j = jnt.nv
        if nv_j == 0:
            continue
        idx_v = jnt.idx_v
        for k in range(nv_j):
            # Pinocchio stores one damping/friction value per DOF of the joint
            damping_vec[idx_v + k]  = model.damping[idx_v + k]
            friction_vec[idx_v + k] = model.friction[idx_v + k]

    # Friction force vectors (evaluated at v=0 for reference;
    # in practice these depend on the actual velocity at runtime):
    #   F_viscous(v) = diag(damping_vec)  · v          [linear in v]
    #   F_dry(v)     = friction_vec * sign(v)           [nonlinear]
    #   F_total(v)   = F_viscous(v) + F_dry(v)
    #
    # At v=0: F_viscous=0, F_dry is undefined (sign(0)=0 by convention here).
    F_viscous_at_v0 = damping_vec  * v           # = 0 since v=0
    F_dry_at_v0     = friction_vec * np.sign(v)  # = 0 since v=0

    # Build the viscous damping matrix D = diag(damping_vec)  (nv × nv)
    D = np.diag(damping_vec)

    # 6. Write output
    output_file = os.path.join(
        os.path.expanduser('~/limo_project/ros2_ws/useful_scripts'),
        'dynamics4matlab_planar.txt'
    )
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    sep  = "=" * 72
    sep2 = "-" * 72

    with open(output_file, 'w', encoding='utf-8') as f:

        # ------------------------------------------------------------------
        # Header
        # ------------------------------------------------------------------
        f.write(f"{sep}\n")
        f.write("DYNAMIC MATRICES — LIMO ACKERMANN  (planar base + wheel joints)\n")
        f.write(f"{sep}\n\n")
        f.write(f"  Root joint  : JointModelPlanar()  — 3 DOF  (x, y, θ_z)\n")
        f.write(f"  Wheel/steer : 4 continuous + 2 revolute  — 6 DOF\n")
        f.write(f"  Total  nq   : {model.nq}   "
                f"(planar needs 4 scalars: x, y, cos θ, sin θ)\n")
        f.write(f"  Total  nv   : {model.nv}   "
                f"(3 planar velocities + 6 joint velocities)\n\n")
        f.write("  Equation of motion:  M(q)·v̇ + C(q,v)·v + g(q) + D·v + f·sign(v) = τ\n")
        f.write("    where  D = diag(damping)   [viscous friction matrix]\n")
        f.write("           f = friction vector  [Coulomb / dry friction]\n\n")

        # ------------------------------------------------------------------
        # Chassis summary
        # ------------------------------------------------------------------
        f.write(f"{sep}\n")
        f.write("CHASSIS PARAMETERS LUMPED INTO THE PLANAR ROOT JOINT\n")
        f.write(f"{sep}\n\n")
        f.write(f"  Source URDF link  : inertial_link  (fixed child of base_link)\n")
        f.write(f"  Total chassis mass: {CHASSIS_MASS_KG} kg\n")
        f.write(f"  Ixx (roll)        : 0.24  kg·m²\n")
        f.write(f"  Iyy (pitch)       : 0.96  kg·m²\n")
        f.write(f"  Izz (yaw)         : {CHASSIS_IZZ}  kg·m²  ← active in planar motion\n\n")
        f.write("  Pinocchio automatically propagates the inertia of ALL fixed-child\n")
        f.write("  links (base_link, inertial_link, laser_link, imu_link, camera_link)\n")
        f.write("  to the first movable ancestor, which is the planar root joint.\n")
        f.write("  Therefore M[0:3, 0:3] reflects the TOTAL lumped inertia of the\n")
        f.write("  chassis assembly, not just the URDF inertial_link values.\n\n")

        # ------------------------------------------------------------------
        # Active joints
        # ------------------------------------------------------------------
        f.write(f"{sep}\n")
        f.write("ACTIVE JOINTS IN THE MODEL\n")
        f.write(f"{sep}\n\n")
        for jnt in joint_map:
            name = jnt["name"]
            desc = JOINT_DESCRIPTIONS.get(name, {})
            f.write(f"  ► {name}\n")
            f.write(f"      Type          : {desc.get('dof_type','n/a')}\n")
            f.write(f"      Axes          : {desc.get('axes', desc.get('axis','n/a'))}\n")
            f.write(f"      Parent link   : {desc.get('parent_link','n/a')}\n")
            f.write(f"      Child link    : {desc.get('child_link','n/a')}\n")
            if "position_xyz" in desc:
                f.write(f"      Position      : {desc['position_xyz']}\n")
            f.write(f"      Mass [kg]     : {desc.get('mass_kg','n/a')}\n")
            f.write(f"      Izz [kg·m²]   : {desc.get('inertia_izz','n/a')}\n")
            f.write(f"      q indices     : {jnt['idx_q']} .. {jnt['idx_q']+jnt['nq']-1}  "
                    f"(nq={jnt['nq']})\n")
            f.write(f"      v indices     : {jnt['idx_v']} .. {jnt['idx_v']+jnt['nv']-1}  "
                    f"(nv={jnt['nv']})\n")
            f.write(f"      Notes         : {desc.get('note','')}\n\n")

        # ------------------------------------------------------------------
        # Vector q
        # ------------------------------------------------------------------
        f.write(f"{sep}\n")
        f.write("VECTOR q — NEUTRAL CONFIGURATION  (nq = {model.nq})\n".replace("{model.nq}", str(model.nq)))
        f.write(f"{sep}\n\n")
        f.write("  The planar root joint stores [x, y, cos(θ), sin(θ)]  →  nq = 4.\n")
        f.write("  Each continuous wheel stores [cos(φ), sin(φ)]         →  nq = 2.\n")
        f.write("  Each revolute steering hinge stores [δ]               →  nq = 1.\n\n")
        f.write(format_q_explained(q, joint_map))
        f.write("\n\n")

        # ------------------------------------------------------------------
        # Vector v
        # ------------------------------------------------------------------
        f.write(f"{sep}\n")
        f.write("VECTOR v — GENERALIZED VELOCITIES (zero)  (nv = {model.nv})\n".replace("{model.nv}", str(model.nv)))
        f.write(f"{sep}\n\n")
        f.write("  The planar root joint velocity is [ẋ, ẏ, ω_z]  expressed in the\n")
        f.write("  WORLD frame (local-world-aligned convention in Pinocchio).\n")
        f.write("  Wheel/hinge velocities are angular rates around their local axes.\n\n")
        f.write(format_v_explained(v, joint_map))
        f.write("\n\n")

        # ------------------------------------------------------------------
        # Mass matrix M
        # ------------------------------------------------------------------
        f.write(f"{sep}\n")
        f.write(f"MASS MATRIX M(q)  [kg / kg·m²]  —  {model.nv}×{model.nv}\n")
        f.write(f"{sep}\n\n")
        f.write("  Structure of M in the planar-base model:\n\n")
        f.write("        [ M_chassis(3×3)  |  M_coupling(3×6) ]\n")
        f.write("    M = [ ----------------+-----------------  ]\n")
        f.write("        [ M_coupling^T    |  M_wheels(6×6)   ]\n\n")
        f.write("  M_chassis  : mass/inertia of the whole chassis body.\n")
        f.write("               Diagonal entries ≈ [total_mass, total_mass, Izz_total].\n")
        f.write("               Off-diagonal entries arise from wheel CoM offsets.\n")
        f.write("  M_coupling : dynamic coupling between chassis motion and wheel spin.\n")
        f.write("               Non-zero if wheel CoM is offset from its rotation axis.\n")
        f.write("  M_wheels   : diagonal, each entry = wheel/hinge inertia around its axis.\n\n")
        f.write(format_matrix_explained(M, joint_map, "M(q)", "kg / kg·m²"))
        f.write("\n\n")

        # ------------------------------------------------------------------
        # Coriolis matrix C
        # ------------------------------------------------------------------
        f.write(f"{sep}\n")
        f.write(f"CORIOLIS MATRIX C(q,v)  [kg·m/s or kg·m²/s]  —  {model.nv}×{model.nv}\n")
        f.write(f"{sep}\n\n")
        f.write("  C(q,v)·v contains centrifugal and Coriolis terms.\n")
        f.write("  With v = 0 all terms vanish by definition.\n")
        f.write("  At non-zero velocity, the planar block (rows/cols 0-2) will show\n")
        f.write("  gyroscopic coupling between chassis yaw rate and wheel spin.\n\n")
        f.write(format_matrix_explained(C, joint_map, "C(q,v)", "kg·m/s or kg·m²/s"))
        f.write("\n\n")

        # ------------------------------------------------------------------
        # Gravity vector g
        # ------------------------------------------------------------------
        f.write(f"{sep}\n")
        f.write(f"GRAVITY VECTOR g(q)  [N or N·m]  —  length {model.nv}\n")
        f.write(f"{sep}\n\n")
        f.write("  For a robot moving on a HORIZONTAL plane, gravity acts along -Z\n")
        f.write("  (world frame).  The planar DOF (x, y, θ_z) are perpendicular to\n")
        f.write("  gravity, so g[0], g[1], g[2] = 0.\n")
        f.write("  Wheel/hinge CoMs lie on their rotation axes, so g[3..8] = 0 too.\n\n")
        f.write(format_gravity_explained(g, joint_map))
        f.write("\n\n")

        # ------------------------------------------------------------------
        # Friction parameters
        # ------------------------------------------------------------------
        f.write(f"{sep}\n")
        f.write("FRICTION PARAMETERS  (from URDF <dynamics> tags)\n")
        f.write(f"{sep}\n\n")
        f.write("  Source URDF tags:\n")
        f.write("    Steering hinges : <dynamics damping=\"1.0\" friction=\"2.0\"/>\n")
        f.write("    Wheels          : no <dynamics> tag → damping=0, friction=0\n")
        f.write("    Planar root     : virtual joint, no friction\n\n")
        f.write("  Complete equation of motion with friction:\n\n")
        f.write("    M(q)·v̇  =  τ  −  C(q,v)·v  −  g(q)  −  D·v  −  f·sign(v)\n\n")
        f.write("  where:\n")
        f.write("    D·v        = viscous friction  (linear in velocity)\n")
        f.write("    f·sign(v)  = Coulomb / dry friction  (nonlinear, direction-dependent)\n\n")

        f.write("  Per-DOF friction breakdown:\n")
        f.write("  " + "-" * 68 + "\n")
        f.write(f"  {'DOF':<4}  {'Joint':<32}  {'damping b [N·m·s/rad]':>22}  {'friction f [N·m]':>18}\n")
        f.write("  " + "-" * 68 + "\n")
        for jnt in joint_map:
            name  = jnt["name"]
            idx_v = jnt["idx_v"]
            nv_j  = jnt["nv"]
            if nv_j == 3:
                # Planar root: 3 DOF, no friction
                dof_labels = ["ẋ  (x vel)", "ẏ  (y vel)", "ω_z (yaw)"]
                for k, lbl in enumerate(dof_labels):
                    f.write(f"  {idx_v+k:<4}  {name+' → '+lbl:<32}  "
                            f"{damping_vec[idx_v+k]:>22.4f}  {friction_vec[idx_v+k]:>18.4f}\n")
            else:
                f.write(f"  {idx_v:<4}  {name:<32}  "
                        f"{damping_vec[idx_v]:>22.4f}  {friction_vec[idx_v]:>18.4f}\n")
        f.write("  " + "-" * 68 + "\n\n")

        f.write("  Physical meaning of each friction term:\n\n")
        f.write("  VISCOUS (D·v):\n")
        f.write("    Models bearing drag and fluid resistance. Proportional to velocity.\n")
        f.write("    In state-space form adds a linear damping term: effectively\n")
        f.write("    increases the denominator of the closed-loop transfer function.\n")
        f.write("    Only non-zero for steering hinges (b = 1.0 N·m·s/rad).\n\n")
        f.write("  COULOMB (f·sign(v)):\n")
        f.write("    Models static/kinetic friction at the joint. Constant magnitude,\n")
        f.write("    direction opposes motion. Creates a dead-zone around v=0.\n")
        f.write("    Only non-zero for steering hinges (f = 2.0 N·m).\n")
        f.write("    NOTE: at v=0 the sign function is discontinuous; in simulation\n")
        f.write("    use a smoothed approximation: sign(v) ≈ tanh(v / v_eps).\n\n")
        f.write("  WHEEL ROLLING FRICTION:\n")
        f.write("    The URDF declares mu1=10, mu2=10 for Gazebo contact simulation,\n")
        f.write("    but these are contact coefficients for the physics engine, NOT\n")
        f.write("    joint-level friction. They are NOT available in model.friction.\n")
        f.write("    To model rolling resistance analytically, add manually:\n")
        f.write("      F_roll = C_rr · m_wheel · g_z · sign(v_wheel)\n")
        f.write("    with C_rr ≈ 0.01..0.05 for rubber on hard floor.\n\n")

        # ------------------------------------------------------------------
        # MATLAB-ready block
        # ------------------------------------------------------------------
        f.write(f"{sep}\n")
        f.write("RAW MATRICES FOR MATLAB (copy-paste)\n")
        f.write(f"{sep}\n\n")
        np.set_printoptions(precision=6, suppress=True, linewidth=200)

        def to_matlab(arr, name):
            if arr.ndim == 1:
                s = np.array2string(arr, separator=', ')
                return f"{name} = [{s}]';\n"
            rows = []
            for row in arr:
                rows.append(", ".join(f"{v:.6f}" for v in row))
            inner = ";\n  ".join(rows)
            return f"{name} = [{inner}];\n"

        f.write(to_matlab(M, "M"))
        f.write(to_matlab(C, "C"))
        f.write(to_matlab(g, "g"))
        f.write(to_matlab(D, "D"))
        f.write(to_matlab(friction_vec, "f_coulomb"))
        f.write("\n% Usage in MATLAB:\n")
        f.write("%   v_dot = M \\ (tau - C*v - g - D*v - f_coulomb.*sign(v));\n")
        f.write("%   For simulation near v=0, replace sign(v) with tanh(v/v_eps),\n")
        f.write("%   e.g. v_eps = 0.01 rad/s to avoid discontinuity.\n\n")
        f.write(f"\n% DOF legend (v vector, length {model.nv}):\n")
        for jnt in joint_map:
            name  = jnt["name"]
            idx_v = jnt["idx_v"]
            nv_j  = jnt["nv"]
            if nv_j == 3:
                for k, desc in enumerate(PLANAR_V_DESCRIPTIONS):
                    f.write(f"%   v({idx_v+k+1}) = {desc}\n")   # MATLAB is 1-indexed
            else:
                desc = JOINT_DESCRIPTIONS.get(name, {})
                f.write(f"%   v({idx_v+1}) = {name}  dθ/dt [rad/s]  axis {desc.get('axes','?')}\n")
        f.write("\n")

    print(f"[SUCCESS] Output saved to: {output_file}")


if __name__ == "__main__":
    main()