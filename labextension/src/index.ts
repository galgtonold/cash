import { JupyterFrontEnd, JupyterFrontEndPlugin } from '@jupyterlab/application';
import { INotebookTracker, NotebookActions, NotebookPanel } from '@jupyterlab/notebook';

const TARGET = 'cash_live_cells';
const DEBOUNCE_MS = 150;

/**
 * Push the notebook's current cell sources to the kernel.
 *
 * cash reads cells it did not execute from the saved .ipynb, so an unsaved edit
 * is invisible to it. This makes the frontend volunteer them.
 *
 * It pushes rather than answering a request because a comm cannot be serviced
 * while a cell is executing. And it flushes on executionScheduled because shell
 * messages are FIFO: a push sent before the execute_request is processed before
 * it, which turns "edit then immediately run" from a race into an ordering
 * guarantee.
 */
function snapshot(panel: NotebookPanel): any[] {
  const model = panel.content.model;
  if (!model) {
    return [];
  }
  const out: any[] = [];
  for (let i = 0; i < model.cells.length; i++) {
    const cell = model.cells.get(i);
    out.push({
      cell_type: cell.type,
      id: cell.id,
      source: cell.sharedModel.getSource()
    });
  }
  return out;
}

const plugin: JupyterFrontEndPlugin<void> = {
  id: 'cash:live-cells',
  autoStart: true,
  requires: [INotebookTracker],
  activate: (app: JupyterFrontEnd, tracker: INotebookTracker) => {
    let seq = 0;
    const comms = new WeakMap<NotebookPanel, any>();

    /**
     * Push a snapshot. `allowOpen` gates whether a MISSING comm may be opened.
     *
     * Only the execution flush passes true, and that is not a tuning choice. A
     * fresh kernel has not run `import cash` yet, and the flush-before-execute
     * ordering GUARANTEES our comm_open is processed before the cell that
     * imports it -- so the very first open is refused ("No such comm target
     * registered"), ipykernel replies comm_close, and JupyterLab disposes the
     * handler. The only recovery is to open again on the NEXT flush, by which
     * time the import has run.
     *
     * Opening on every debounced keystroke instead would spray that refusal
     * through the kernel log of every user who never imports cash, and buy
     * nothing: the debounced push is a latency optimisation, while the flush is
     * the correctness path -- cash only reads cells while a cell is executing,
     * and the flush always precedes that.
     */
    const send = (panel: NotebookPanel, allowOpen: boolean) => {
      try {
        const kernel = panel.sessionContext?.session?.kernel;
        if (!kernel) {
          return;
        }
        let comm = comms.get(panel);
        if (!comm) {
          if (!allowOpen) {
            return;
          }
          comm = kernel.createComm(TARGET);
          // LOAD-BEARING, not a tuning knob. JupyterLab 4.6 defaults
          // `commsOverSubshells: perCommTarget`, which delivers comms on a
          // SUBSHELL THREAD. That breaks the entire premise of this design:
          // ordering against the execute_request stops being FIFO and becomes a
          // race -- measured as a 0.4-7.4ms lead we merely happened to win,
          // 130 times out of 130, which is exactly the kind of evidence that
          // looks like a guarantee until it is not. Forcing the comm onto the
          // main shell restores true FIFO (verified 56/56).
          //
          // It is also what lets src/cash/notebook/live_cells.py keep a
          // lock-free dict: on the main shell the handler runs on the kernel's
          // own thread. Removing this line silently makes that store
          // cross-thread mutable state.
          //
          // Removing it is caught by scripts/check-comms-over-subshells.js
          // (run by `npm run build`, over BOTH this source and the built
          // bundle) and by tests/test_notebook/test_labextension_packaging.py.
          // Do not delete those guards to "fix" a failure here.
          comm.commsOverSubshells = 'disabled';
          // Without this the extension is MUTE for the life of any kernel that
          // refused the first open -- which, per the note above, is EVERY fresh
          // kernel. `send` would hold the disposed handler forever and every
          // later push would throw into the catch below, silently. Dropping it
          // here lets the next flush build a working one.
          //
          // The identity check stops a late close for an already-replaced comm
          // from evicting its successor.
          const opened = comm;
          opened.onClose = () => {
            if (comms.get(panel) === opened) {
              comms.delete(panel);
            }
          };
          comm.open({});
          comms.set(panel, comm);
        }
        seq += 1;
        comm.send({ seq, cells: snapshot(panel) });
      } catch (err) {
        // A diagnostic aid must never break the frontend. Drop the comm as well
        // as swallowing the error: a disposed handler raises 'Cannot send', and
        // latching on to it here would reproduce exactly the muteness the
        // onClose hook above exists to prevent.
        comms.delete(panel);
        console.debug('[cash] push failed', err);
      }
    };

    tracker.widgetAdded.connect((_, panel) => {
      let timer: number | undefined;
      panel.content.model?.contentChanged.connect(() => {
        window.clearTimeout(timer);
        timer = window.setTimeout(() => send(panel, false), DEBOUNCE_MS);
      });
      // A kernel restart drops the comm. Do NOT rebuild it here: the restarted
      // kernel has also lost its `import cash`, so an immediate re-open would
      // only be refused again. Drop the handle and let the next execution flush
      // open one, once the import has had a chance to run.
      panel.sessionContext.kernelChanged.connect(() => {
        comms.delete(panel);
      });
    });

    // THE ordering guarantee. Send synchronously here, before the
    // execute_request leaves, so FIFO delivers it to the kernel first. Also the
    // only route allowed to OPEN a comm -- see the note on `send`.
    NotebookActions.executionScheduled.connect((_, args) => {
      const panel = tracker.find(p => p.content === args.notebook);
      if (panel) {
        send(panel, true);
      }
    });
  }
};

export default plugin;
