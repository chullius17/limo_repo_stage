import numpy as np
import skfmm
from scipy.ndimage import gaussian_filter


def gradient_descent_backtrack(phi, goal, start, step_size=0.3, max_steps=2000):
    """
    Gradient descent su phi dal goal verso lo start.
    Robusto ai bordi: se il punto corrente è sul bordo, il passo viene
    clampato all'interno dell'array invece di restituire None.
    """
    phi_smooth = gaussian_filter(phi.astype(np.float64), sigma=1.5)
    grad_y, grad_x = np.gradient(phi_smooth)

    rows, cols = phi.shape
    path       = [goal]
    current    = np.array(goal, dtype=float)
    start_arr  = np.array(start, dtype=float)

    for _ in range(max_steps):
        r = int(round(current[0]))
        c = int(round(current[1]))

        # Clamp invece di uscire: se siamo sul bordo rimaniamo dentro
        r = int(np.clip(r, 0, rows - 1))
        c = int(np.clip(c, 0, cols - 1))

        # Convergenza
        if np.linalg.norm(current - start_arr) < 1.5:
            path.append(start)
            return path

        gy = grad_y[r, c]
        gx = grad_x[r, c]

        # Fallback su intorno 3x3 se gradiente nullo/NaN
        if np.isnan(gy) or np.isnan(gx) or (abs(gy) < 1e-9 and abs(gx) < 1e-9):
            gy, gx = _neighborhood_gradient(phi_smooth, r, c)
            if gy is None:
                return None

        norm = np.hypot(gx, gy)
        if norm < 1e-9:
            return None

        # Passo in discesa
        new_r = current[0] - step_size * (gy / norm)
        new_c = current[1] - step_size * (gx / norm)

        # Clamp entro i bounds — non usciamo mai dall'array
        current[0] = np.clip(new_r, 0, rows - 1)
        current[1] = np.clip(new_c, 0, cols - 1)

        new_point = (int(round(current[0])), int(round(current[1])))
        if new_point != path[-1]:
            path.append(new_point)

        # Stallo: se restiamo nello stesso pixel per 10 step consecutivi usciamo
        if len(path) >= 10 and len(set(path[-10:])) == 1:
            return None

    return None


def _neighborhood_gradient(phi, r, c):
    """Gradiente finito su intorno 3x3, ignora celle inf/NaN."""
    rows, cols = phi.shape
    gy_list, gx_list = [], []

    for dr in range(-1, 2):
        for dc in range(-1, 2):
            if dr == 0 and dc == 0:
                continue
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                v = phi[nr, nc]
                v0 = phi[r, c]
                if np.isfinite(v) and np.isfinite(v0):
                    gy_list.append((v - v0) * dr)
                    gx_list.append((v - v0) * dc)

    if not gy_list:
        return None, None
    return float(np.mean(gy_list)), float(np.mean(gx_list))


def fmm_multi_goal(cost_map, start, goals, max_cost_threshold=np.inf):
    """
    Fast Marching Method multi-goal con penalizzazione delle manovre basse.
    Propaga il fronte d'onda da `start` e traccia i percorsi verso ogni goal feasible.
    """
    rows, cols = cost_map.shape

    # Start sicuro: mai sul bordo assoluto
    sr = int(np.clip(start[0], 1, rows - 2))
    sc = int(np.clip(start[1], 1, cols - 2))
    start_safe = (sr, sc)

    # ── PENALIZZAZIONE VERTICALE DELLA ROI ────────────────────────────────────
    # Creiamo un profilo lineare: 0.0 in cima (r=0) fino a un valore massimo in fondo (r=rows-1).
    # Usiamo una penalità additiva per forzare il fronte ad avanzare dritto prima di piegare.
    # Puoi regolare il valore 40.0 (più è alto, più i percorsi tenderanno a stare alti).
    vertical_penalty = np.linspace(0.0, 0.0, rows, dtype=np.float32)
    
    # Trasformiamo il vettore in una matrice (rows, cols) tramite broadcasting
    penalty_matrix = vertical_penalty[:, np.newaxis]
    
    # Applichiamo la penalità alla mappa dei costi originale
    cost_map_penalized = cost_map + penalty_matrix
    # ──────────────────────────────────────────────────────────────────────────

    # Speed map: zone a costo 0 hanno velocità 1, ostacoli rallentano la propagazione
    finite_thr = max_cost_threshold if np.isfinite(max_cost_threshold) else 1e6
    cost_clipped = np.clip(cost_map_penalized, 0.0, finite_thr)
    speed_map = 1.0 / (1.0 + cost_clipped)

    # Maschera skfmm (0 = sorgente, 1 = tutto il resto)
    march_mask = np.ones((rows, cols), dtype=np.int32)
    march_mask[start_safe] = 0

    phi_masked = skfmm.travel_time(march_mask, speed=speed_map)
    phi = np.ma.filled(phi_masked, np.inf)

    # Backtracking
    feasible_paths = {}
    thr = finite_thr

    for i, goal in enumerate(goals):
        gr, gc = goal
        if not (0 <= gr < rows and 0 <= gc < cols):
            print(f"Goal {i+1} {goal}: UNFEASIBLE (fuori bounds)")
            continue

        val = phi[goal]
        if np.isinf(val) or np.isnan(val) or val > thr:
            print(f"Goal {i+1} {goal}: UNFEASIBLE (irraggiungibile, phi={val:.1f})")
            continue

        path = gradient_descent_backtrack(phi, goal, start_safe)

        if path is not None:
            feasible_paths[goal] = path
            print(f"Goal {i+1} {goal}: FEASIBLE ({len(path)} punti, phi={val:.1f})")
        else:
            print(f"Goal {i+1} {goal}: UNFEASIBLE (gradient tracking failure, phi={val:.1f})")

    return feasible_paths


# ── Test ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_map = np.zeros((100, 100))
    test_map[0:30, 90:100] = 99999.0

    start_point = (95, 50)
    goal_list = [(5, 20), (5, 50), (5, 80), (10, 95)]

    valid = fmm_multi_goal(test_map, start_point, goal_list, max_cost_threshold=5000.0)
    print(f"\nRisultato: {len(goal_list)} goal → {len(valid)} percorsi validi.")