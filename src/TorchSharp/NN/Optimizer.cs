// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using static TorchSharp.torch;

namespace TorchSharp
{
    using Modules;

    public static partial class torch
    {
        public static partial class optim
        {
            public partial class Optimizer : IDisposable
            {
                /// <summary>
                ///    Class wrapping PyTorch's optimzer object reference.
                /// </summary>
                internal sealed class HType : SafeHandle
                {
                    public HType(IntPtr preexistingHandle, bool ownsHandle) : base(IntPtr.Zero, ownsHandle)
                    {
                        SetHandle(preexistingHandle);
                    }

                    public override bool IsInvalid => handle == IntPtr.Zero;

                    // This is just for marshalling
                    internal HType() : base(IntPtr.Zero, true)
                    {
                    }

                    [DllImport("LibTorchSharp")]
                    private static extern void THSNN_Optimizer_dispose(HType handle);

                    protected override bool ReleaseHandle()
                    {
                        THSNN_Optimizer_dispose(this);
                        return true;
                    }

                    protected override void Dispose(bool disposing)
                    {
                        if (disposing) {
                            ReleaseHandle();
                        }
                    }
                }

                internal HType handle;

                protected Optimizer(IntPtr handle)
                {
                    if (handle != IntPtr.Zero) {
                        this.handle = new HType(handle, true);
                    }
                }

                ~Optimizer()
                {
                    Dispose(false);
                }

                /// <summary>
                ///   Releases the storage.
                /// </summary>
                public void Dispose()
                {
                    Dispose(true);
                    GC.SuppressFinalize(this);
                }

                /// <summary>
                ///   Implements the .NET Dispose pattern.
                /// </summary>
                protected virtual void Dispose(bool disposing)
                {
                    if (disposing && handle != null && !handle.IsInvalid) {
                        handle.Dispose();
                        handle.SetHandleAsInvalid();
                    }
                }

                [DllImport("LibTorchSharp")]
                private static extern void THSNN_Optimizer_zero_grad(HType module);

                public virtual void zero_grad()
                {
                    THSNN_Optimizer_zero_grad(handle);
                    torch.CheckForErrors();
                }

                [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
                public delegate IntPtr LossClosure();


                [DllImport("LibTorchSharp")]
                private static extern IntPtr THSNN_Optimizer_step(HType module, LossClosure closure);

                public virtual Tensor step(Func<Tensor> closure = null)
                {
                    IntPtr res = (closure == null) ?
                        THSNN_Optimizer_step(handle, null) :
                        THSNN_Optimizer_step(handle, () => {
                            return closure().DecoupleFromNativeHandle();
                        });

                    if (res == IntPtr.Zero)
                        torch.CheckForErrors();

                    return (res == IntPtr.Zero) ? null : new Tensor(res);
                }

                [DllImport("LibTorchSharp")]
                private static extern void THSNN_Optimizer_getParameters(HType module, AllocatePinnedArray allocator);

                public IEnumerable<Tensor> parameters()
                {
                    IntPtr[] ptrArray;

                    using (var pa = new PinnedArray<IntPtr>()) {
                        THSNN_Optimizer_getParameters(handle, pa.CreateArray);
                        torch.CheckForErrors();
                        ptrArray = pa.Array;
                    }
                    return ptrArray.Select(x => new Tensor(x));
                }
            }

            public interface ILearningRateController
            {
                double LearningRate { set; get; }

                double InitialLearningRate { set; get; }
            }

            public interface IMomentum
            {
                double Momentum { get; set; }
            }

            public interface IBetas
            {
                (double, double) Betas { get; set; }
            }

            [DllImport("LibTorchSharp")]
            private static extern IntPtr THSNN_LBFGS_ctor(IntPtr parameters, int len, double learningRate, long max_iter, long max_eval, double tolerange_grad, double tolerance_change, long history_size);

            public static LBFGSOptimizer LBFGS(IEnumerable<Tensor> parameters, double learningRate = 0.01, long max_iter = 20, long? max_eval = null, double tolerange_grad = 1e-5, double tolerance_change = 1e-9, long history_size = 100)
            {
                if (!max_eval.HasValue) max_eval = 5 * max_iter / 4;

                var parray = new PinnedArray<IntPtr>();
                IntPtr paramsRef = parray.CreateArray(parameters.Select(p => p.Handle).ToArray());

                var res = THSNN_LBFGS_ctor(paramsRef, parray.Array.Length, learningRate, max_iter, max_eval.Value, tolerange_grad, tolerance_change, history_size);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new LBFGSOptimizer(res, learningRate);
            }

            [DllImport("LibTorchSharp")]
            private static extern IntPtr THSNN_RMSprop_ctor(IntPtr parameters, int len, double learningRate, double alpha, double eps, double weight_decay, double momemtum, bool centered);

            /// <summary>
            /// Implements RMSprop algorithm.
            ///
            /// Proposed by G.Hinton in his course.
            /// </summary>
            /// <param name="parameters">Parameters to optimize</param>
            /// <param name="learningRate">Learning rate (default: 1e-2)</param>
            /// <param name="alpha">Smoothing constant (default: 0.99)</param>
            /// <param name="eps">Term added to the denominator to improve numerical stability (default: 1e-8)</param>
            /// <param name="weight_decay">Weight decay (L2 penalty) (default: 0)</param>
            /// <param name="momentum">Momentum factor (default: 0)</param>
            /// <param name="centered">if true, compute the centered RMSProp, the gradient is normalized by an estimation of its variance</param>
            /// <returns></returns>
            public static RMSPropOptimizer RMSProp(IEnumerable<Tensor> parameters, double learningRate = 0.01, double alpha = 0.99, double eps = 1e-8, double weight_decay = 0, double momentum = 0, bool centered = false)
            {
                var parray = new PinnedArray<IntPtr>();
                IntPtr paramsRef = parray.CreateArray(parameters.Select(p => p.Handle).ToArray());

                var res = THSNN_RMSprop_ctor(paramsRef, parray.Array.Length, learningRate, alpha, eps, weight_decay, momentum, centered);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new RMSPropOptimizer(res, learningRate, momentum);
            }

            [DllImport("LibTorchSharp")]
            private static extern IntPtr THSNN_Adam_ctor(IntPtr parameters, int len, double learningRate, double beta1, double beta2, double eps, double weight_decay, bool amsgrad);

            /// <summary>
            /// Implements Adam algorithm.
            ///
            /// It has been proposed in Adam: A Method for Stochastic Optimization.The implementation of the L2 penalty follows changes proposed in Decoupled Weight Decay Regularization.
            /// </summary>
            /// <param name="parameters">Parameters to optimize</param>
            /// <param name="learningRate">learning rate (default: 1e-3)</param>
            /// <param name="beta1">Coefficient used for computing running averages of gradient and its square (default: 0.9)</param>
            /// <param name="beta2">Coefficient used for computing running averages of gradient and its square (default: 0.999)</param>
            /// <param name="eps">Term added to the denominator to improve numerical stability (default: 1e-8)</param>
            /// <param name="weight_decay">Weight decay (L2 penalty) (default: 0)</param>
            /// <param name="amsgrad">Whether to use the AMSGrad variant of this algorithm. (default: False)</param>
            /// <returns></returns>
            public static AdamOptimizer Adam(IEnumerable<Tensor> parameters, double learningRate = 1e-3, double beta1 = 0.9, double beta2 = 0.99, double eps = 1e-8, double weight_decay = 0, bool amsgrad = false)
            {
                var parray = new PinnedArray<IntPtr>();
                IntPtr paramsRef = parray.CreateArray(parameters.Select(p => p.Handle).ToArray());

                var res = THSNN_Adam_ctor(paramsRef, parray.Array.Length, learningRate, beta1, beta2, eps, weight_decay, amsgrad);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new AdamOptimizer(res, learningRate, beta1, beta2);
            }

            [DllImport("LibTorchSharp")]
            private static extern IntPtr THSNN_AdamW_ctor(IntPtr parameters, int len, double learningRate, double beta1, double beta2, double eps, double weight_decay, bool amsgrad);

            /// <summary>
            /// Implements Adam algorithm.
            ///
            /// It has been proposed in Adam: A Method for Stochastic Optimization. The AdamW variant was proposed in Decoupled Weight Decay Regularization.
            /// </summary>
            /// <param name="parameters">Parameters to optimize</param>
            /// <param name="learningRate">learning rate (default: 1e-3)</param>
            /// <param name="beta1">Coefficient used for computing running averages of gradient and its square (default: 0.9)</param>
            /// <param name="beta2">Coefficient used for computing running averages of gradient and its square (default: 0.999)</param>
            /// <param name="eps">Term added to the denominator to improve numerical stability (default: 1e-8)</param>
            /// <param name="weight_decay">Weight decay (L2 penalty) (default: 0)</param>
            /// <param name="amsgrad">Whether to use the AMSGrad variant of this algorithm. (default: False)</param>
            /// <returns></returns>
            public static AdamWOptimizer AdamW(IEnumerable<Tensor> parameters, double learningRate = 1e-3, double beta1 = 0.9, double beta2 = 0.99, double eps = 1e-8, double weight_decay = 0, bool amsgrad = false)
            {
                var parray = new PinnedArray<IntPtr>();
                IntPtr paramsRef = parray.CreateArray(parameters.Select(p => p.Handle).ToArray());

                var res = THSNN_AdamW_ctor(paramsRef, parray.Array.Length, learningRate, beta1, beta2, eps, weight_decay, amsgrad);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new AdamWOptimizer(res, learningRate, beta1, beta2);
            }

            [DllImport("LibTorchSharp")]
            private static extern IntPtr THSNN_Adagrad_ctor(IntPtr parameters, int len, double learningRate, double lr_decay, double weight_decay, double initial_accumulator_value, double eps);

            /// <summary>
            /// Implements Adagrad algorithm.
            ///
            /// It has been proposed in Adaptive Subgradient Methods for Online Learning and Stochastic Optimization.
            /// </summary>
            /// <param name="parameters">Parameters to optimize</param>
            /// <param name="learningRate">learning rate (default: 1e-2)</param>
            /// <param name="lr_decay">learning rate decay (default: 0)</param>
            /// <param name="weight_decay">weight decay (L2 penalty) (default: 0)</param>
            /// <param name="initial_accumulator_value"></param>
            /// <param name="eps">Term added to the denominator to improve numerical stability (default: 1e-10)</param>
            /// <returns></returns>
            public static AdagradOptimizer Adagrad(IEnumerable<Tensor> parameters, double learningRate = 1e-2, double lr_decay = 0, double weight_decay = 0, double initial_accumulator_value = 0, double eps = 1e-10)
            {
                var parray = new PinnedArray<IntPtr>();
                IntPtr paramsRef = parray.CreateArray(parameters.Select(p => p.Handle).ToArray());

                var res = THSNN_Adagrad_ctor(paramsRef, parray.Array.Length, learningRate, lr_decay, weight_decay, initial_accumulator_value, eps);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new AdagradOptimizer(res, learningRate);
            }

            /// <summary>
            /// Implements Adadelta algorithm.
            ///
            /// It has been proposed in ADADELTA: An Adaptive Learning Rate Method.
            /// </summary>
            /// <param name="named_parameters">Parameters to optimize. This optimizer requires the <b>named</b> parameters collection.</param>
            /// <param name="lr ">Learning rate</param>
            /// <param name="rho">Coefficient used for computing a running average of squared gradients (default: 0.9)</param>
            /// <param name="eps">Term added to the denominator to improve numerical stability, i.e. avoid division-by-zero (default: 1e-6)</param>
            /// <param name="weight_decay">Weight decay (L2 penalty) (default: 0)</param>
            /// <returns></returns>
            public static AdadeltaOptimizer Adadelta(IEnumerable<(string name, Modules.Parameter parameter)> named_parameters, double lr = 1.0, double rho = 0.9, double eps = 1e-6, double weight_decay = 0)
            {
                return new AdadeltaOptimizer(named_parameters, lr, rho, eps, weight_decay);
            }

            /// <summary>
            /// Implements NAdam algorithm.
            ///
            /// For further details regarding the algorithm we refer to Incorporating Nesterov Momentum into Adam.
            /// https://openreview.net/forum?id=OM0jvwB8jIp57ZJjtNEZ
            /// </summary>
            /// <param name="named_parameters">Parameters to optimize. This optimizer requires the <b>named</b> parameters collection.</param>
            /// <param name="lr ">Learning rate</param>
            /// <param name="beta1">Coefficient used for computing running averages of gradient and its square (default: 0.9)</param>
            /// <param name="beta2">Coefficient used for computing running averages of gradient and its square (default: 0.999)</param>
            /// <param name="eps">Term added to the denominator to improve numerical stability, i.e. avoid division-by-zero (default: 1e-8)</param>
            /// <param name="weight_decay">Weight decay (L2 penalty) (default: 0)</param>
            /// <param name="momentum_decay">Momentum decay</param>
            public static NAdamOptimizer NAdam(IEnumerable<(string name, Modules.Parameter parameter)> named_parameters, double lr = 0.002, double beta1 = 0.9, double beta2 = 0.999, double eps = 1e-8, double weight_decay = 0, double momentum_decay = 4e-3)
            {
                return new NAdamOptimizer(named_parameters, lr, beta1, beta2, eps, weight_decay, momentum_decay);
            }

            /// <summary>
            /// Implements RAdam algorithm.
            ///
            /// For further details regarding the algorithm we refer to 'On the variance of the adaptive learning rate and beyond.'
            /// https://arxiv.org/abs/1908.03265
            /// </summary>
            /// <param name="named_parameters">Parameters to optimize. This optimizer requires the <b>named</b> parameters collection.</param>
            /// <param name="lr ">Learning rate</param>
            /// <param name="beta1">Coefficient used for computing running averages of gradient and its square (default: 0.9)</param>
            /// <param name="beta2">Coefficient used for computing running averages of gradient and its square (default: 0.999)</param>
            /// <param name="eps">Term added to the denominator to improve numerical stability, i.e. avoid division-by-zero (default: 1e-8)</param>
            /// <param name="weight_decay">Weight decay (L2 penalty) (default: 0)</param>
            public static RAdamOptimizer RAdam(IEnumerable<(string name, Modules.Parameter parameter)> named_parameters, double lr = 0.002, double beta1 = 0.9, double beta2 = 0.999, double eps = 1e-8, double weight_decay = 0)
            {
                return new RAdamOptimizer(named_parameters, lr, beta1, beta2, eps, weight_decay);
            }

            /// <summary>
            /// Implements Adamax algorithm (a variant of Adam based on infinity norm).
            ///
            /// It has been proposed in Adam: A Method for Stochastic Optimization.
            /// </summary>
            /// <param name="named_parameters">Parameters to optimize. This optimizer requires the <b>named</b> parameters collection.</param>
            /// <param name="lr ">Learning rate</param>
            /// <param name="beta1">Coefficient used for computing running averages of gradient and its square (default: 0.9)</param>
            /// <param name="beta2">Coefficient used for computing running averages of gradient and its square (default: 0.999)</param>
            /// <param name="eps">Term added to the denominator to improve numerical stability, i.e. avoid division-by-zero (default: 1e-8)</param>
            /// <param name="weight_decay">Weight decay (L2 penalty) (default: 0)</param>
            /// <returns></returns>
            public static AdamaxOptimizer Adamax(IEnumerable<(string name, Modules.Parameter parameter)> named_parameters, double lr = 0.002, double beta1 = 0.9, double beta2 = 0.999, double eps = 1e-8, double weight_decay = 0)
            {
                return new AdamaxOptimizer(named_parameters, lr, beta1, beta2, eps, weight_decay);
            }

            /// <summary>
            /// Implements Averaged Stochastic Gradient Descent.
            ///
            /// It has been proposed in Acceleration of stochastic approximation by averaging.
            /// </summary>
            /// <param name="named_parameters">Parameters to optimize. This optimizer requires the <b>named</b> parameters collection.</param>
            /// <param name="lr ">Learning rate</param>
            /// <param name="lambd">Decay term (default: 1e-4)</param>
            /// <param name="alpha">Power for eta update (default: 0.75)</param>
            /// <param name="t0">Point at which to start averaging (default: 1e6)</param>
            /// <param name="weight_decay">Weight decay (L2 penalty) (default: 0)</param>
            /// <returns></returns>
            public static ASGDOptimizer ASGD(IEnumerable<(string name, Modules.Parameter parameter)> named_parameters, double lr = 1e-3, double lambd = 1e-4, double alpha = 0.75, double t0 = 1e6, double weight_decay = 0)
            {
                return new ASGDOptimizer(named_parameters, lr, lambd, alpha, t0, weight_decay);
            }

            /// <summary>
            /// Implements the resilient backpropagation algorithm.
            /// </summary>
            /// <param name="named_parameters">Parameters to optimize. This optimizer requires the <b>named</b> parameters collection.</param>
            /// <param name="lr ">Learning rate</param>
            /// <param name="etaminus">Multiplicative increase factor.</param>
            /// <param name="etaplus">Multiplicative decrease factor.</param>
            /// <param name="min_step">Minimum allowed step size.</param>
            /// <param name="max_step">Maximum allowed step size.</param>
            /// <returns></returns>
            public static RpropOptimizer Rprop(IEnumerable<(string name, Modules.Parameter parameter)> named_parameters, double lr = 1e-2, double etaminus = 0.5, double etaplus = 1.2, double min_step = 1e-6, double max_step = 50)
            {
                return new RpropOptimizer(named_parameters, lr, etaminus, etaplus, min_step, max_step);
            }

            [DllImport("LibTorchSharp")]
            private static extern IntPtr THSNN_SGD_ctor(IntPtr parameters, int len, double learningRate, double momentum, double dampening, double weight_decay, bool nesterov);

            /// <summary>
            /// Implements stochastic gradient descent (optionally with momentum).
            /// </summary>
            /// <param name="parameters">Parameters to optimize</param>
            /// <param name="learningRate">Learning rate</param>
            /// <param name="momentum">Momentum factor (default: 0)</param>
            /// <param name="dampening">Dampening for momentum (default: 0)</param>
            /// <param name="weight_decay">Weight decay (L2 penalty) (default: 0)</param>
            /// <param name="nesterov">Enables Nesterov momentum (default: False)</param>
            /// <returns></returns>
            public static SGDOptimizer SGD(IEnumerable<Tensor> parameters, double learningRate, double momentum = 0, double dampening = 0, double weight_decay = 0, bool nesterov = false)
            {
                var parray = new PinnedArray<IntPtr>();
                IntPtr paramsRef = parray.CreateArray(parameters.Select(p => p.Handle).ToArray());

                var res = THSNN_SGD_ctor(paramsRef, parray.Array.Length, learningRate, momentum, dampening, weight_decay, nesterov);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new SGDOptimizer(res, learningRate, momentum);
            }
        }
    }

    namespace Modules
    {
        using static torch.optim;

        // Most optimizers are implemented in native code, but a few of them are directly implemented in
        // managed code.

        /// <summary>
        /// Base class to help with a couple of the things that managed-code implementations need.
        /// </summary>
        public class OptimizerHelper : Optimizer, ILearningRateController
        {
            public OptimizerHelper(IEnumerable<(string name, Modules.Parameter parameter)> named_parameters, double learningRate) : base(IntPtr.Zero)
            {
                LearningRate = learningRate;
                InitialLearningRate = learningRate;
                _parameters = named_parameters;
            }

            public override void zero_grad()
            {
                foreach (var (_, p) in _parameters) {

                    using var grad = p.grad();

                    if (grad is null) continue;

                    grad.zero_().Dispose();
                }
            }

            public double LearningRate { get; set; }

            public double InitialLearningRate { get; set; }

            protected IEnumerable<(string name, Modules.Parameter parameter)> _parameters;
        }

        public class AdadeltaOptimizer : OptimizerHelper, ILearningRateController
        {
            /// <summary>
            /// Constructor
            /// </summary>
            /// <param name="named_parameters">Parameters to optimize. This optimizer requires the <b>named</b> parameters collection.</param>
            /// <param name="lr ">Learning rate</param>
            /// <param name="rho">Coefficient used for computing a running average of squared gradients (default: 0.9)</param>
            /// <param name="eps">Term added to the denominator to improve numerical stability, i.e. avoid division-by-zero (default: 1e-6)</param>
            /// <param name="weight_decay">Weight decay (L2 penalty) (default: 0)</param>
            public AdadeltaOptimizer(IEnumerable<(string name, Modules.Parameter parameter)> named_parameters, double lr = 1.0, double rho = 0.9, double eps = 1e-6, double weight_decay = 0) : base(named_parameters, lr)
            {
                LearningRate = lr;
                InitialLearningRate = lr;
                _rho = rho;
                _eps = eps;
                _weight_decay = weight_decay;

                foreach (var (name, p) in named_parameters) {
                    var state = new State();
                    _state[name] = state;
                    state.step = 0;
                    state.square_avg = torch.zeros_like(p);
                    state.acc_delta = torch.zeros_like(p);
                }
            }

            public override Tensor step(Func<Tensor> closure = null)
            {
                Tensor loss = null;

                if (closure != null) {
                    loss = closure();
                }

                using (var _ = torch.no_grad()) {

                    using (var d = torch.NewDisposeScope()) {

                        foreach (var (name, param) in _parameters) {

                            var grad = param.grad();

                            if (grad is null) continue;

                            if (grad.is_sparse) throw new ArgumentException("Adadelta does not support sparse gradients");

                            var state = _state[name];

                            var square_avg = state.square_avg;
                            var acc_delta = state.acc_delta;

                            grad = (_weight_decay != 0)
                                ? grad.add(param, alpha: _weight_decay)
                                : grad.alias();

                            square_avg.mul_(_rho).addcmul_(grad, grad, 1 - _rho);

                            var std = square_avg.add(_eps).sqrt_();
                            var delta = acc_delta.add(_eps).sqrt_().div_(std).mul_(grad);

                            param.add_(delta, alpha: -LearningRate);
                            acc_delta.mul_(_rho).addcmul_(delta, delta, 1 - _rho);
                        }

                        d.DisposeEverything();
                    }
                }

                return loss;
            }

            protected override void Dispose(bool disposing)
            {
                base.Dispose(disposing);
                foreach (var (name, state) in _state) {
                    state.square_avg.Dispose();
                    state.acc_delta.Dispose();
                }
            }

            private class State
            {
                public int step;
                public Tensor square_avg;
                public Tensor acc_delta;
            }

            private Dictionary<string, State> _state = new Dictionary<string, State>();
            private double _rho;
            private double _eps;
            private double _weight_decay;
        }

        public class AdamaxOptimizer : OptimizerHelper, ILearningRateController, IBetas
        {
            /// <summary>
            /// Implements Adamax algorithm (a variant of Adam based on infinity norm).
            ///
            /// It has been proposed in Adam: A Method for Stochastic Optimization.
            /// </summary>
            /// <param name="named_parameters">Parameters to optimize. This optimizer requires the <b>named</b> parameters collection.</param>
            /// <param name="lr ">Learning rate</param>
            /// <param name="beta1">Coefficient used for computing running averages of gradient and its square (default: 0.9)</param>
            /// <param name="beta2">Coefficient used for computing running averages of gradient and its square (default: 0.999)</param>
            /// <param name="eps">Term added to the denominator to improve numerical stability, i.e. avoid division-by-zero (default: 1e-8)</param>
            /// <param name="weight_decay">Weight decay (L2 penalty) (default: 0)</param>
            /// <returns></returns>
            public AdamaxOptimizer(IEnumerable<(string name, Modules.Parameter parameter)> named_parameters, double lr = 0.002, double beta1 = 0.9, double beta2 = 0.999, double eps = 1e-8, double weight_decay = 0) : base(named_parameters, lr)
            {
                LearningRate = lr;
                InitialLearningRate = lr;
                _beta1 = beta1;
                _beta2 = beta2;
                _eps = eps;
                _weight_decay = weight_decay;

                foreach (var (name, p) in named_parameters) {
                    var state = new State();
                    _state[name] = state;
                    state.step = 0;
                    state.exp_avg = torch.zeros_like(p);
                    state.exp_inf = torch.zeros_like(p);
                }
            }

            public override Tensor step(Func<Tensor> closure = null)
            {
                Tensor loss = null;

                if (closure != null) {
                    loss = closure();
                }

                using (var _ = torch.no_grad()) {

                    using (var d = torch.NewDisposeScope()) {

                        foreach (var (name, param) in _parameters) {

                            var grad = param.grad();

                            if (grad is null) continue;

                            if (grad.is_sparse) throw new ArgumentException("Adamax does not support sparse gradients");

                            var state = _state[name];

                            state.step += 1;

                            var exp_avg = state.exp_avg;
                            var exp_inf = state.exp_inf;

                            grad = (_weight_decay != 0)
                                ? grad.add(param, alpha: _weight_decay)
                                : grad.alias();

                            exp_avg.mul_(_beta1).add_(grad, alpha: 1 - _beta1);

                            var norm_buf = torch.cat(new Tensor[] {
                            exp_inf.mul_(_beta2).unsqueeze(0),
                            grad.abs().add_(_eps).unsqueeze_(0) }, 0);

                            torch.amax(norm_buf, new long[] { 0 }, false, exp_inf);

                            var clr = LearningRate / (1 - Math.Pow(_beta1, state.step));
                            param.addcdiv_(exp_avg, exp_inf, value: -clr);
                        }

                        d.DisposeEverything();
                    }
                }

                return loss;
            }

            protected override void Dispose(bool disposing)
            {
                base.Dispose(disposing);
                foreach (var (name, state) in _state) {
                    state.exp_avg.Dispose();
                    state.exp_inf.Dispose();
                }
            }

            public (double, double) Betas {
                get => (_beta1, _beta2);
                set { _beta1 = value.Item1; _beta2 = value.Item2; }
            }

            private class State
            {
                public int step;
                public Tensor exp_avg;
                public Tensor exp_inf;
            }

            private Dictionary<string, State> _state = new Dictionary<string, State>();
            private double _beta1;
            private double _beta2;
            private double _eps;
            private double _weight_decay;
        }

        public class NAdamOptimizer : OptimizerHelper, ILearningRateController, IBetas
        {
            /// <summary>
            /// Implements NAdam algorithm.
            ///
            /// For further details regarding the algorithm we refer to Incorporating Nesterov Momentum into Adam.
            /// https://openreview.net/forum?id=OM0jvwB8jIp57ZJjtNEZ
            /// </summary>
            /// <param name="named_parameters">Parameters to optimize. This optimizer requires the <b>named</b> parameters collection.</param>
            /// <param name="lr ">Learning rate</param>
            /// <param name="beta1">Coefficient used for computing running averages of gradient and its square (default: 0.9)</param>
            /// <param name="beta2">Coefficient used for computing running averages of gradient and its square (default: 0.999)</param>
            /// <param name="eps">Term added to the denominator to improve numerical stability, i.e. avoid division-by-zero (default: 1e-8)</param>
            /// <param name="weight_decay">Weight decay (L2 penalty) (default: 0)</param>
            /// <param name="momentum_decay">Momentum decay</param>
            /// <returns></returns>
            public NAdamOptimizer(IEnumerable<(string name, Modules.Parameter parameter)> named_parameters, double lr = 0.002, double beta1 = 0.9, double beta2 = 0.999, double eps = 1e-8, double weight_decay = 0, double momentum_decay = 4e-3) : base(named_parameters, lr)
            {
                LearningRate = lr;
                InitialLearningRate = lr;
                _beta1 = beta1;
                _beta2 = beta2;
                _eps = eps;
                _weight_decay = weight_decay;
                _momentum_decay = momentum_decay;

                foreach (var (name, p) in named_parameters) {
                    var state = new State();
                    _state[name] = state;
                    state.step = 0;
                    state.mu_product = 1;
                    state.exp_avg = torch.zeros_like(p);
                    state.exp_avg_sq = torch.zeros_like(p);
                }
            }

            public override Tensor step(Func<Tensor> closure = null)
            {
                Tensor loss = null;

                if (closure != null) {
                    loss = closure();
                }

                using (var _ = torch.no_grad()) {

                    using (var d = torch.NewDisposeScope()) {

                        foreach (var (name, param) in _parameters) {

                            var grad = param.grad();

                            if (grad is null) continue;

                            var state = _state[name];

                            state.step += 1;

                            var exp_avg = state.exp_avg;
                            var exp_avg_sq = state.exp_avg_sq;

                            var bias_correction2 = 1 - Math.Pow(_beta2, state.step);

                            grad = (_weight_decay != 0)
                                ? grad.add(param, alpha: _weight_decay)
                                : grad.alias();

                            var mu = _beta1 * (1.0 - 0.5 * Math.Pow(0.96, state.step * _momentum_decay));
                            var mu_next = _beta1 * (1.0 - 0.5 * Math.Pow(0.96, (state.step + 1) * _momentum_decay));

                            var mu_product = state.mu_product * mu;
                            var mu_product_next = mu_product * mu * mu_next;

                            exp_avg.mul_(_beta1).add_(grad, alpha: 1 - _beta1);
                            exp_avg_sq.mul_(_beta2).addcmul_(grad, grad, value: 1 - _beta2);

                            var denom = exp_avg_sq.div(bias_correction2).sqrt_().add_(_eps);

                            param.addcdiv_(grad, denom, value: -LearningRate * (1 - mu) / (1 - mu_product));
                            param.addcdiv_(exp_avg, denom, value: -LearningRate * mu_next / (1 - mu_product_next));

                            state.mu_product = mu_product;
                        }

                        d.DisposeEverything();
                    }
                }

                return loss;
            }

            protected override void Dispose(bool disposing)
            {
                base.Dispose(disposing);
                foreach (var (name, state) in _state) {
                    state.exp_avg.Dispose();
                    state.exp_avg_sq.Dispose();
                }
            }

            public (double, double) Betas {
                get => (_beta1, _beta2);
                set { _beta1 = value.Item1; _beta2 = value.Item2; }
            }

            private class State
            {
                public int step;
                public double mu_product;
                public Tensor exp_avg;
                public Tensor exp_avg_sq;
            }

            private Dictionary<string, State> _state = new Dictionary<string, State>();
            private double _beta1;
            private double _beta2;
            private double _eps;
            private double _weight_decay;
            private double _momentum_decay;
        }

        public class RAdamOptimizer : OptimizerHelper, ILearningRateController, IBetas
        {
            /// <summary>
            /// Implements RAdam algorithm.
            ///
            /// For further details regarding the algorithm we refer to 'On the variance of the adaptive learning rate and beyond.'
            /// https://arxiv.org/abs/1908.03265
            /// </summary>
            /// <param name="named_parameters">Parameters to optimize. This optimizer requires the <b>named</b> parameters collection.</param>
            /// <param name="lr ">Learning rate</param>
            /// <param name="beta1">Coefficient used for computing running averages of gradient and its square (default: 0.9)</param>
            /// <param name="beta2">Coefficient used for computing running averages of gradient and its square (default: 0.999)</param>
            /// <param name="eps">Term added to the denominator to improve numerical stability, i.e. avoid division-by-zero (default: 1e-8)</param>
            /// <param name="weight_decay">Weight decay (L2 penalty) (default: 0)</param>
            /// <returns></returns>
            public RAdamOptimizer(IEnumerable<(string name, Modules.Parameter parameter)> named_parameters, double lr = 0.002, double beta1 = 0.9, double beta2 = 0.999, double eps = 1e-8, double weight_decay = 0) : base(named_parameters, lr)
            {
                LearningRate = lr;
                InitialLearningRate = lr;
                _beta1 = beta1;
                _beta2 = beta2;
                _eps = eps;
                _weight_decay = weight_decay;

                foreach (var (name, p) in named_parameters) {
                    var state = new State();
                    _state[name] = state;
                    state.step = 0;
                    state.exp_avg = torch.zeros_like(p);
                    state.exp_avg_sq = torch.zeros_like(p);
                }
            }

            public override Tensor step(Func<Tensor> closure = null)
            {
                Tensor loss = null;

                if (closure != null) {
                    loss = closure();
                }

                using (var _ = torch.no_grad()) {

                    using (var d = torch.NewDisposeScope()) {

                        foreach (var (name, param) in _parameters) {

                            var grad = param.grad();

                            if (grad is null) continue;

                            var state = _state[name];

                            state.step += 1;

                            var exp_avg = state.exp_avg;
                            var exp_avg_sq = state.exp_avg_sq;

                            var bias_correction1 = 1 - Math.Pow(_beta1, state.step);
                            var bias_correction2 = 1 - Math.Pow(_beta2, state.step);

                            grad = (_weight_decay != 0)
                                ? grad.add(param, alpha: _weight_decay)
                                : grad.alias();

                            exp_avg.mul_(_beta1).add_(grad, alpha: 1 - _beta1);
                            exp_avg_sq.mul_(_beta2).addcmul_(grad, grad, value: 1 - _beta2);

                            var bias_corrected_exp_avg = exp_avg / bias_correction1;

                            var rho_inf = 2 / (1 - _beta2) - 1;
                            var rho_t = rho_inf - 2 * state.step * Math.Pow(_beta2, state.step) / bias_correction2;

                            var t6 = bias_corrected_exp_avg * LearningRate;

                            if (rho_t > 5) {
                                var rect = Math.Sqrt((rho_t - 4) * (rho_t - 2) * rho_inf / ((rho_inf - 4) * (rho_inf - 2) * rho_t));
                                var adaptive_lr = Math.Sqrt(bias_correction2) / exp_avg_sq.sqrt().add_(_eps);

                                param.add_(t6 * LearningRate * adaptive_lr * rect, alpha: -1.0);
                            } else {
                                param.add_(t6, alpha: -1.0);
                            }
                        }

                        d.DisposeEverything();
                    }
                }

                return loss;
            }

            protected override void Dispose(bool disposing)
            {
                base.Dispose(disposing);
                foreach (var (name, state) in _state) {
                    state.exp_avg.Dispose();
                    state.exp_avg_sq.Dispose();
                }
            }

            private class State
            {
                public int step;
                public Tensor exp_avg;
                public Tensor exp_avg_sq;
            }

            private Dictionary<string, State> _state = new Dictionary<string, State>();
            private double _beta1;
            private double _beta2;
            private double _eps;
            private double _weight_decay;

            public (double, double) Betas {
                get => (_beta1, _beta2);
                set { _beta1 = value.Item1; _beta2 = value.Item2; }
            }
        }

        public class ASGDOptimizer : OptimizerHelper, ILearningRateController
        {
            /// <summary>
            /// Implements ASGD algorithm (a variant of Adam based on infinity norm).
            ///
            /// It has been proposed in Adam: A Method for Stochastic Optimization.
            /// </summary>
            /// <param name="named_parameters">Parameters to optimize. This optimizer requires the <b>named</b> parameters collection.</param>
            /// <param name="lr ">Learning rate</param>
            /// <param name="lambd">Decay term (default: 1e-4)</param>
            /// <param name="alpha">Power for eta update (default: 0.75)</param>
            /// <param name="t0">Point at which to start averaging (default: 1e6)</param>
            /// <param name="weight_decay">Weight decay (L2 penalty) (default: 0)</param>
            /// <returns></returns>
            public ASGDOptimizer(IEnumerable<(string name, Modules.Parameter parameter)> named_parameters, double lr = 1e-3, double lambd = 1e-4, double alpha = 0.75, double t0 = 1e6, double weight_decay = 0) : base(named_parameters, lr)
            {
                LearningRate = lr;
                InitialLearningRate = lr;
                _lambd = lambd;
                _alpha = alpha;
                _t0 = t0;
                _weight_decay = weight_decay;

                foreach (var (name, p) in named_parameters) {
                    var state = new State();
                    _state[name] = state;
                    state.step = 0;
                    state.eta = lr;
                    state.mu = 1;
                    state.ax = torch.zeros_like(p);
                }
            }

            public override Tensor step(Func<Tensor> closure = null)
            {
                Tensor loss = null;

                if (closure != null) {
                    loss = closure();
                }

                using (var _ = torch.no_grad()) {

                    using (var d = torch.NewDisposeScope()) {

                        foreach (var (name, param) in _parameters) {

                            var grad = param.grad();

                            if (grad is null) continue;

                            if (grad.is_sparse) throw new ArgumentException("ASGD does not support sparse gradients");

                            var state = _state[name];

                            state.step += 1;

                            grad = (_weight_decay != 0)
                                ? grad.add(param, alpha: _weight_decay)
                                : grad.alias();

                            param.mul_(1 - _lambd * state.eta);
                            param.add_(grad, alpha: -state.eta);

                            if (state.mu != 1) {
                                state.ax.add_(param.sub(state.ax).mul(state.mu));
                            } else {
                                state.ax.copy_(param);
                            }

                            state.eta = LearningRate / Math.Pow((1 + _lambd * LearningRate * state.step), _alpha);
                            state.mu = 1 / Math.Max(1, state.step - _t0);
                        }

                        d.DisposeEverything();
                    }
                }

                return loss;
            }

            protected override void Dispose(bool disposing)
            {
                base.Dispose(disposing);
                foreach (var (name, state) in _state) {
                    state.ax.Dispose();
                }
            }

            private class State
            {
                public int step;
                public double eta;
                public double mu;
                public Tensor ax;
            }

            private Dictionary<string, State> _state = new Dictionary<string, State>();
            private double _lambd;
            private double _alpha;
            private double _t0;
            private double _weight_decay;
        }

        public class RpropOptimizer : OptimizerHelper, ILearningRateController
        {
            /// <summary>
            /// Implements Rprop algorithm (a variant of Adam based on infinity norm).
            ///
            /// It has been proposed in Adam: A Method for Stochastic Optimization.
            /// </summary>
            /// <param name="named_parameters">Parameters to optimize. This optimizer requires the <b>named</b> parameters collection.</param>
            /// <param name="lr ">Learning rate</param>
            /// <param name="etaminus">Multiplicative increase factor.</param>
            /// <param name="etaplus">Multiplicative decrease factor.</param>
            /// <param name="min_step">Minimum allowed step size.</param>
            /// <param name="max_step">Maximum allowed step size.</param>
            /// <returns></returns>
            public RpropOptimizer(IEnumerable<(string name, Modules.Parameter parameter)> named_parameters, double lr = 1e-2, double etaminus = 0.5, double etaplus = 1.2, double min_step = 1e-6, double max_step = 50) : base(named_parameters, lr)
            {
                LearningRate = lr;
                InitialLearningRate = lr;
                _etaminus = etaminus;
                _etaplus = etaplus;
                _min_step = min_step;
                _max_step = max_step;

                foreach (var (name, p) in _parameters) {
                    var state = new State();
                    _state[name] = state;
                    state.step = 0;
                    state.prev = torch.zeros_like(p);
                    state.step_size = p.new_empty(p.shape).fill_(LearningRate);
                }
            }

            public override Tensor step(Func<Tensor> closure = null)
            {
                Tensor loss = null;

                if (closure != null) {
                    loss = closure();
                }

                using (var _ = torch.no_grad()) {

                    using (var d = torch.NewDisposeScope()) {

                        foreach (var (name, p) in _parameters) {

                            var grad = p.grad();

                            if (grad is null) continue;

                            if (grad.is_sparse) throw new ArgumentException("Rprop does not support sparse gradients");

                            var state = _state[name];

                            state.step += 1;

                            grad = (_max_step != 0)
                                ? grad.add(p, alpha: _max_step)
                                : grad.alias();

                            var sign = grad.mul(state.prev).sign();
                            sign[sign.gt(0)] = (Tensor)_etaplus;
                            sign[sign.lt(0)] = (Tensor)_etaminus;
                            sign[sign.eq(0)] = (Tensor)1;

                            state.step_size.mul_(sign).clamp_(_min_step, _max_step);

                            grad = grad.clone();

                            grad.index_put_(0, sign.eq(_etaminus));

                            p.addcmul_(grad.sign(), state.step_size, -1);

                            state.prev.copy_(grad);
                        }

                        d.DisposeEverything();
                    }
                }

                return loss;
            }

            protected override void Dispose(bool disposing)
            {
                base.Dispose(disposing);
                foreach (var (name, state) in _state) {
                    state.prev.Dispose();
                    state.step_size.Dispose();
                }
            }

            private class State
            {
                public int step;
                public Tensor prev;
                public Tensor step_size;
            }

            private Dictionary<string, State> _state = new Dictionary<string, State>();
            private double _etaminus;
            private double _etaplus;
            private double _min_step;
            private double _max_step;
        }

        // The following optimizers are just wrappers for the native code implementations.

        public class AdagradOptimizer : Optimizer, ILearningRateController
        {
            public AdagradOptimizer(IntPtr handle, double lr) : base(handle)
            {
                _rate = lr;
                InitialLearningRate = lr;
            }

            [DllImport("LibTorchSharp")]
            private static extern void THSNN_Adagrad_set_lr(HType optimizer, double lr);

            public double LearningRate {
                get { return _rate; }
                set { THSNN_Adagrad_set_lr(handle, value); torch.CheckForErrors(); _rate = value; }
            }

            public double InitialLearningRate { get; set; }

            private double _rate;
        }

        public class AdamOptimizer : Optimizer, ILearningRateController, IBetas
        {
            public AdamOptimizer(IntPtr handle, double lr, double beta1, double beta2) : base(handle)
            {
                _betas = (beta1, beta2);
                _rate = lr;
                InitialLearningRate = lr;
            }

            [DllImport("LibTorchSharp")]
            private static extern void THSNN_Adam_set_lr(HType optimizer, double lr);

            [DllImport("LibTorchSharp")]
            private static extern void THSNN_Adam_set_betas(HType optimizer, double beta1, double beta2);

            public double LearningRate {
                get { return _rate; }
                set { THSNN_Adam_set_lr(handle, value); torch.CheckForErrors(); _rate = value; }
            }

            public double InitialLearningRate { get; set; }

            public (double, double) Betas {
                get => _betas;
                set { THSNN_Adam_set_betas(handle, value.Item1, value.Item2); torch.CheckForErrors(); _betas = value; }
            }
            private (double, double) _betas;

            private double _rate;
        }

        public class AdamWOptimizer : Optimizer, ILearningRateController, IBetas
        {
            public AdamWOptimizer(IntPtr handle, double lr, double beta1, double beta2) : base(handle)
            {
                _betas = (beta1, beta2);
                _rate = lr;
                InitialLearningRate = lr;
            }

            [DllImport("LibTorchSharp")]
            private static extern void THSNN_AdamW_set_lr(HType optimizer, double lr);

            [DllImport("LibTorchSharp")]
            private static extern void THSNN_AdamW_set_betas(HType optimizer, double beta1, double beta2);

            public double LearningRate {
                get { return _rate; }
                set { THSNN_AdamW_set_lr(handle, value); torch.CheckForErrors(); _rate = value; }
            }

            public double InitialLearningRate { get; set; }

            public (double, double) Betas {
                get => _betas;
                set { THSNN_AdamW_set_betas(handle, value.Item1, value.Item2); torch.CheckForErrors(); _betas = value; }
            }
            private (double, double) _betas;

            private double _rate;

        }

        public class RMSPropOptimizer : Optimizer, ILearningRateController, IMomentum
        {
            public RMSPropOptimizer(IntPtr handle, double lr, double momentum) : base(handle)
            {
                _rate = lr;
                _momentum = momentum;
                InitialLearningRate = lr;
            }

            [DllImport("LibTorchSharp")]
            private static extern void THSNN_RMSprop_set_lr(HType optimizer, double lr);

            [DllImport("LibTorchSharp")]
            private static extern void THSNN_RMSprop_set_momentum(HType optimizer, double momentum);

            public double LearningRate {
                get { return _rate; }
                set { THSNN_RMSprop_set_lr(handle, value); torch.CheckForErrors(); _rate = value; }
            }

            public double InitialLearningRate { get; set; }

            private double _rate;

            public double Momentum {
                get => _momentum;
                set { THSNN_RMSprop_set_momentum(handle, value); torch.CheckForErrors(); _momentum = value; }
            }

            private double _momentum;
        }

        public class LBFGSOptimizer : Optimizer, ILearningRateController
        {
            public LBFGSOptimizer(IntPtr handle, double lr) : base(handle)
            {
                _rate = lr;
                InitialLearningRate = lr;
            }

            [DllImport("LibTorchSharp")]
            private static extern void THSNN_LBFGS_set_lr(HType optimizer, double lr);

            public double LearningRate {
                get { return _rate; }
                set { THSNN_LBFGS_set_lr(handle, value); torch.CheckForErrors(); _rate = value; }
            }

            public double InitialLearningRate { get; set; }

            public override Tensor step(Func<Tensor> closure = null)
            {
                if (closure == null)
                    throw new ArgumentNullException("'closure' must be non-null when using the LBFGS optimizer. See: https://pytorch.org/docs/1.9.0/optim.html");
                return base.step(closure);
            }

            private double _rate;
        }

        public class SGDOptimizer : Optimizer, ILearningRateController, IMomentum
        {
            public SGDOptimizer(IntPtr handle, double lr, double momentum) : base(handle)
            {
                _rate = lr;
                _momentum = momentum;
                InitialLearningRate = lr;
            }

            [DllImport("LibTorchSharp")]
            private static extern void THSNN_SGD_set_lr(HType optimizer, double lr);

            [DllImport("LibTorchSharp")]
            private static extern void THSNN_SGD_set_momentum(HType optimizer, double momentum);

            public double LearningRate {
                get { return _rate; }
                set { THSNN_SGD_set_lr(handle, value); torch.CheckForErrors(); _rate = value; }
            }

            public double InitialLearningRate { get; set; }

            private double _rate;

            public double Momentum {
                get => _momentum;
                set { THSNN_SGD_set_momentum(handle, value); torch.CheckForErrors(); _momentum = value; }
            }

            private double _momentum;
        }
    }
}

