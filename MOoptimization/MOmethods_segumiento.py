from collections import Counter
import numpy as np
def log_full(generacion, num_evaluaciones, fitness, hv0, state, dup_child, diags):
    #salidas = {}
    if diags:
        LSruns = len(diags)
        salidas = dict(Counter(d["msg_salida"] for d in diags))
        d_ok   = sum(d["drop_ok"] for d in diags)
        d_disp = sum(d.get("drop_disp", 0) for d in diags)
        d_ev   = sum(d["ev_drop"] for d in diags)
        s_ok   = sum(d["swap_ok"] for d in diags)
        s_disp = sum(d["swap_pares_disp"] for d in diags)
        s_ev   = sum(d["ev_swap"] for d in diags)
        i_ok   = sum(d["inter_ok"] for d in diags)
        i_disp = sum(d["inter_pares_disp"] for d in diags)
        i_ev   = sum(d["ev_inter"] for d in diags)
        gasto_ls = d_ev + s_ev + i_ev
        ls = (
            f"LSruns={LSruns}, "
            f"DROP ok={d_ok},disp={d_disp},evs={d_ev}, "
            f"SWAP ok={s_ok},disp={s_disp},evs={s_ev}, "
            f"INTER ok={i_ok},disp={i_disp},evs={i_ev}, "
            f"gastoLS={gasto_ls}"
        )
    else:
        ls = "sin-LS"
    print(f"Gen={generacion:5d} evals={num_evaluaciones:7d}, "
          f"f1={float(np.min(fitness[:,0])):.8f}, f2={int(np.min(fitness[:,1])):4d}, "
          f"HV={hv0:.6f}, noImpHV={state['count_no_improve_hv0']:4d}, dup={dup_child:.2f}|{ls}")








def log_full_(generacion, num_evaluaciones, fitness, hv0, state, dup_child, diags):
    if diags:
        LSruns = len(diags)
        salidas = dict(Counter(d["msg_salida"] for d in diags))
        d_ok   = sum(d["drop_ok"] for d in diags)
        d_disp = sum(d.get("drop_disp", 0) for d in diags)
        d_ev   = sum(d["ev_drop"] for d in diags)
        s_ok   = sum(d["swap_ok"] for d in diags)
        s_disp = sum(d["swap_pares_disp"] for d in diags)
        s_ev   = sum(d["ev_swap"] for d in diags)
        i_ok   = sum(d["inter_ok"] for d in diags)
        i_disp = sum(d["inter_pares_disp"] for d in diags)
        i_ev   = sum(d["ev_inter"] for d in diags)
        gasto_ls = d_ev + s_ev + i_ev
        ls = (f"LSruns={LSruns}, oks/disp-evals => "
            f"DROP {d_ok}/{d_disp} - {d_ev}, "
            f"SWAP {s_ok}/{s_disp} - {s_ev}, "
            f"INTER {i_ok}/{i_disp} - {i_ev},"
            f"gastoLS={gasto_ls}" )
    else:
        ls = "sin-LS"
    #extra_salidas = f" | salidas={salidas}" if diags else ""
    '''
    print(f"Gen={generacion:5d} evals={num_evaluaciones:7d}, "
          f"f1={float(np.min(fitness[:,0])):.8f}, f2={int(np.min(fitness[:,1])):4d}, "  #muestra los mínimos de forma independiente. 
          f"HV={hv0:.6f}, noImprovHV={state['count_no_improve_hv0']:4d}, dupChild={dup_child:.2f} |{ls}")
    '''
    # se cambia para que se muestre: la solución con menor error error y sus número de genes de la población, así como también
    #                                la solución con menos gnes y el error que tiene
    idx_best_f1 = int(np.argmin(fitness[:, 0]))
    idx_best_f2 = int(np.argmin(fitness[:, 1]))
    print(f"Gen={generacion:5d} evals={num_evaluaciones:7d}, "
        f"solbestErr=(err={float(fitness[idx_best_f1,0]):.8f}=> su #genes={int(fitness[idx_best_f1,1])}), "
        f"solbestSize=(err={float(fitness[idx_best_f2,0]):.8f}, #genes={int(fitness[idx_best_f2,1])}), "
        f"HV={hv0:.6f}, noImpHV={state['count_no_improve_hv0']:4d}, dup={dup_child:.2f}|{ls}")