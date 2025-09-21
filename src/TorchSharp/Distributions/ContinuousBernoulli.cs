using System;

#nullable enable
namespace TorchSharp.Distributions
{
    public static partial class torch
    {
        public static partial class distributions
        {
            public abstract class ContinuousBernoulli : TorchSharp.torch.distributions.ExponentialFamily
            {
                public ContinuousBernoulli(TorchSharp.torch.Tensor probs, TorchSharp.torch.Tensor logits, (float, float)? lims, TorchSharp.torch.Generator? generator=null) : base(generator)
                {
                    if (!lims.HasValue)
                        lims = (0.499f, 0.501f);
                    //https://github.com/pytorch/pytorch/blob/v2.8.0/torch/distributions/continuous_bernoulli.py#L23
                    throw new NotImplementedException();
                }
            }
        }
    }
    
}
