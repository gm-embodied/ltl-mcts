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
        return p > off_th  # Fix: strictly greater than off_th to keep ON state
    else:
        return p >= on_th  # Greater than or equal to on_th to switch to ON state


def update_window_state(ws: WindowState, pA: float, pB: float, W: int,
                        on_th: float, off_th: float):
    """
    Update windowed continuous surveillance state.
    
    Fix "death trap" logic:
    1. Delay arming: Only arm when the target is first detected (ap becomes ON)
    2. After arming, reset timer to 0 when ap is ON
    3. After arming, ap is OFF, the timer accumulates
    4. When the timer reaches the window limit, trigger violation
    5. Unarmed state will not trigger violation

    Return: New state, whether to trigger resetA/resetB, whether to violate
    """
    newA_on = hysteresis(ws.apA_on, pA, on_th, off_th)
    newB_on = hysteresis(ws.apB_on, pB, on_th, off_th)

    resetA = False
    resetB = False
    
    # Fix death trap: delay arming, only arm when the target is first detected
    if not ws.armed_A and newA_on:
        ws.armed_A = True
    if not ws.armed_B and newB_on:
        ws.armed_B = True

    # Region A logic
    if newA_on:
        if not ws.apA_on:
            resetA = True
            ws.confirm_count_A += 1
        ws.t_A = 0  # Reset timer when ap is ON
    else:
        # Only accumulate timer when ap is OFF after arming
        if ws.armed_A:
            ws.t_A = min(ws.t_A + 1, W)

    # Region B logic
    if newB_on:
        if not ws.apB_on:
            resetB = True
            ws.confirm_count_B += 1
        ws.t_B = 0  # Reset timer when ap is ON
    else:
        # Only accumulate timer when ap is OFF after arming
        if ws.armed_B:
            ws.t_B = min(ws.t_B + 1, W)

    ws.apA_on = newA_on
    ws.apB_on = newB_on

    # Violation condition: only armed regions can violate
    violate = ((ws.armed_A and ws.t_A >= W) or (ws.armed_B and ws.t_B >= W))
    return ws, resetA, resetB, violate


