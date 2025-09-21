using System;

namespace TorchSharp.Utils
{
    /// <summary>
    /// Specifies the custom name for a component to be used in the module's state_dict instead of the default field name.
    /// </summary>
    public class ComponentNameAttribute : Attribute
    {
        public string Name { get; set; }
    }
}
