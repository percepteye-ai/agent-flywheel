/**
 * The two failures a caller must be able to tell apart.
 *
 * `LeaseLost` is not an error in the operational sense -- it is the control
 * plane saying "this rollout is no longer yours", which happens whenever a
 * lease lapses and the work is requeued. A caller that treats it as a
 * transport failure retries forever against a rollout someone else now owns.
 */
export class TransportError extends Error {
  constructor(message, options) {
    super(message, options);
    this.name = "TransportError";
  }
}

export class LeaseLost extends TransportError {
  constructor(message, options) {
    super(message, options);
    this.name = "LeaseLost";
  }
}
