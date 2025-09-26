from dataclasses import dataclass


@dataclass
class WindowState:
    t_A: int
    t_B: int
    apA_on: bool
    apB_on: bool
    confirm_count_A: int
    confirm_count_B: int
    armed_A: bool = False
    armed_B: bool = False


def hysteresis(prev_on: bool, p: float, on_th: float, off_th: float) -> bool:
    if prev_on:
        return p > off_th  # fix: strictly greater than off_th to maintain ON state
    else:
        return p >= on_th  # greater than or equal to on_th to switch to ON state


def update_window_state(ws: WindowState, pA: float, pB: float, W: int,
                        on_th: float, off_th: float):
    """
    Update windowed persistent surveillance state.
    
    Fixed "death trap" logic:
    1. Delayed arming: only arm when first detecting target (ap becomes ON)
    2. After armed, reset timer to 0 when ap is ON
    3. After armed, accumulate timer when ap is OFF
    4. Trigger violation when timer reaches window limit
    5. No violation in unarmed state

    Returns: new state, whether to trigger resetA/resetB, whether violation occurred
    """
    newA_on = hysteresis(ws.apA_on, pA, on_th, off_th)
    newB_on = hysteresis(ws.apB_on, pB, on_th, off_th)

    resetA = False
    resetB = False
    
    # Fix death trap: delayed arming, only arm when first detecting target
    if not ws.armed_A and newA_on:
        ws.armed_A = True
    if not ws.armed_B and newB_on:
        ws.armed_B = True

    # Region A logic
    if newA_on:
        if not ws.apA_on:
            resetA = True
            ws.confirm_count_A += 1
        ws.t_A = 0  # reset timer when ap is ON
    else:
        # only accumulate timer when armed and ap is OFF
        if ws.armed_A:
            ws.t_A = min(ws.t_A + 1, W)

    # Region B logic
    if newB_on:
        if not ws.apB_on:
            resetB = True
            ws.confirm_count_B += 1
        ws.t_B = 0  # reset timer when ap is ON
    else:
        # only accumulate timer when armed and ap is OFF
        if ws.armed_B:
            ws.t_B = min(ws.t_B + 1, W)

    ws.apA_on = newA_on
    ws.apB_on = newB_on

    # Violation condition: only armed regions can violate
    violate = ((ws.armed_A and ws.t_A >= W) or (ws.armed_B and ws.t_B >= W))
    return ws, resetA, resetB, violate


