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

    const send = (panel: NotebookPanel) => {
      try {
        const kernel = panel.sessionContext?.session?.kernel;
        if (!kernel) {
          return;
        }
        let comm = comms.get(panel);
        if (!comm) {
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
          comm.open({});
          comms.set(panel, comm);
        }
        seq += 1;
        comm.send({ seq, cells: snapshot(panel) });
      } catch (err) {
        // A diagnostic aid must never break the frontend.
        console.debug('[cash] push failed', err);
      }
    };

    tracker.widgetAdded.connect((_, panel) => {
      let timer: number | undefined;
      panel.content.model?.contentChanged.connect(() => {
        window.clearTimeout(timer);
        timer = window.setTimeout(() => send(panel), DEBOUNCE_MS);
      });
      // A kernel restart drops the comm; rebuild it and resend.
      panel.sessionContext.kernelChanged.connect(() => {
        comms.delete(panel);
        send(panel);
      });
    });

    // THE ordering guarantee. Send synchronously here, before the
    // execute_request leaves, so FIFO delivers it to the kernel first.
    NotebookActions.executionScheduled.connect((_, args) => {
      const panel = tracker.find(p => p.content === args.notebook);
      if (panel) {
        send(panel);
      }
    });
  }
};

export default plugin;
