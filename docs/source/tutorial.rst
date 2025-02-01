Tutorial
=====

Overview
--------

Trapezoidal integration
--------

.. code-block:: python

    import torch  # Required package for torchint
    import torchint
    
    data_type = torch.float64
    device_type = 'cuda'
    torchint.set_backend(data_type, device_type) # This sets single precision data type, and device in the backend
    
    def function (x):
        return torch.sin(x)
    
    bound = [[0, 1]] # This sets integral limitation as (0,1).
    num_point = [20] # This sets number of sampling points per dimension.
    integral_value = torchint.trapz_integrate(function, None, bound, num_point, None) #We use trapz_integrate function
    
    analytical_value = torch.cos(torch.tensor(0, device=device_type, dtype=data_type))-\
                        torch.cos(torch.tensor(1, device=device_type, dtype=data_type))  # absolute value of this integral
    relative_error = torch.abs(integral_value - analytical_value) / analytical_value # relative error
    
    print(f"integral value: {integral_value.item():.10f}") # Convert to Python float
    print(f"analytical value: {analytical_value.item():.10f}")
    print(f"relative error: {relative_error.item():.10%}")
    
    print(integral_value.dtype)
    print(integral_value.device)

.. code-block:: None

    integral value: 0.4595915725
    analytical value: 0.4596976941
    relative error: 0.0230850917%
    torch.float64
    cuda:0



.. code-block:: python

    import torch  # Required package for torchint
    import torchint
    
    data_type = torch.float64
    device_type = 'cuda'
    torchint.set_backend(data_type, device_type) # This sets single precision data type, and device in the backend
    
    def function(x1, x2, x3, params): # this is the standard way to define an integrand with parameters
        a1 = params[0]
        a2 = params[1]
        a3 = params[2]
        return a1 * torch.exp(-a2 * (x1**2 + x2**2 + x3**2)) + a3 * torch.sin(x1) * torch.cos(x2) * torch.exp(x3)
    
    # This sets the parameter set, which is a 2d array in all cases. In this case, we have 1e4 parameter sets
    a1_values = torch.linspace(1.0, 10.0, 10000, dtype = data_type, device = device_type)
    a2_values = torch.linspace(2.0, 20.0, 10000, dtype = data_type, device = device_type)
    a3_values = torch.linspace(0.5, 5, 10000, dtype = data_type, device = device_type)
    param_values = torch.stack((a1_values, a2_values, a3_values), dim=1)
    
    bound = [[0, 1], [0, 1], [0, 1]] # This sets integral limitation as (0,1),(0,1), and (0,1) for x1, x2, and x3, respectively.
    num_point = [20, 20, 20] # This sets number of sampling points per dimension.
    
    def boundary(x1, x2, x3):
        condition1 = x1**2 + x2**2 + x3**2 > 0.2
        condition2 = x1**2 + x2**2 + x3**2 < 0.8
        return condition1 & condition2
    
    integral_value = torchint.trapz_integrate(function, param_values, bound, num_point, boundary) # We use trapz_integrate function
    
    print(f"integral value: {integral_value}") # Output integral value
    print(f"length of integral value: {integral_value.size()}") # Output length of the integral value
    
    # To estimate error, we double the grids in all three dimension, and output the relative error.
    num_point = [40, 40, 40] # This sets number of sampling points per dimension, which are doubled
    integral_value2 = torchint.trapz_integrate(function, param_values, bound, num_point, boundary) #We use trapz_integrate function
    relative_error = torch.abs(integral_value - integral_value2) / integral_value # relative error
    
    print(f"integral value with denser grids: {integral_value2}")
    print(f"relative error: {relative_error}")
    
    print(integral_value.dtype)
    print(integral_value.device)

.. code-block:: None

    integral value: tensor([0.1923, 0.1924, 0.1925,  ..., 0.7314, 0.7315, 0.7315], device='cuda:0',
           dtype=torch.float64)
    length of integral value: torch.Size([10000])
    integral value with denser grids: tensor([0.1935, 0.1936, 0.1937,  ..., 0.7386, 0.7387, 0.7387], device='cuda:0',
           dtype=torch.float64)
    relative error: tensor([0.0062, 0.0062, 0.0062,  ..., 0.0098, 0.0098, 0.0098], device='cuda:0',
           dtype=torch.float64)
    torch.float64
    cuda:0


Simpson's integration
--------

.. code-block:: python


.. code-block:: None


.. code-block:: python


.. code-block:: None









Boole's integration
--------





.. code-block:: python


.. code-block:: None


.. code-block:: python


.. code-block:: None




Gaussian quadrature
--------


.. code-block:: python


.. code-block:: None


.. code-block:: python


.. code-block:: None




Monte Carlo integration
--------

.. code-block:: python


.. code-block:: None


.. code-block:: python


.. code-block:: None




















