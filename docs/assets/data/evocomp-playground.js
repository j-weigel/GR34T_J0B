window.NOTEBOOKS = window.NOTEBOOKS || {};
window.NOTEBOOKS["evocomp-playground"] = {
 "name": "EvoCompPlayground.ipynb",
 "kind": "notebook",
 "download": "assets/notebooks/EvoCompPlayground.ipynb",
 "cells": [
  {
   "type": "markdown",
   "source": "## Pumpkin Problem (relocated):"
  },
  {
   "type": "code",
   "source": "import numpy as np\nimport matplotlib.pyplot as plt\n\n# Define our pumpkin function\ndef f(x):\n    return 10 - (x - 5)**2 # \ud835\udc53(\ud835\udc65)=10\u2212(\ud835\udc65\u22125)^2\n\n# Define the derivative (needed for Newton and Gradient Descent)\n# f'(x) = -2(x - 2)\ndef df(x):\n    return -2 * (x - 5) # Corrected to use multiplication operator\n\n# Define the second derivative (needed for Newton's Method)\n# f''(x) = -2\ndef ddf(x):\n    return -2\n\n# Let's visualize the pumpkin's flight first!\nx_vals = np.linspace(0,10, 100)\nplt.plot(x_vals, f(x_vals), label='Pumpkin Path')\nplt.axvline(5, color='red', linestyle='--', label='Actual Peak (x=5)')\nplt.title(\"Pumpkin Trajectory\")\nplt.xlabel(\"Distance (m)\")\nplt.ylabel(\"Height (m)\")\nplt.legend()\nplt.grid(True)\nplt.show()\n\n\n# REFLECTION \u2014\n# Take-home:\n# Changing the surface changes the location of the optimum, but the\n# algorithm itself does not need to change. We only give it the new\n# function, derivatives, and account for where the new optima will be.",
   "outputs": [
    {
     "kind": "image",
     "src": "assets/img/evocomp-playground/cell-01-0.png",
     "width": 574,
     "height": 455
    }
   ]
  },
  {
   "type": "markdown",
   "source": "## Multiple Peaks / Gradient Ascent :"
  },
  {
   "type": "code",
   "source": "# Explore what happens when a surface has MORE THAN ONE peak.\n#\n# New surface:\n#\n#     f(x) = -(x + 2)^2 + 5 + 0.7 * sin(2x)\n#\n# The sinusoidal term creates bumps in the surface.\n# This means the surface is no longer a simple parabola.\n#\n# For this session, we calculate the derivatives analytically.\ndef session_2_surface(x):\n    return -(x + 2) ** 2 + 5 + 1.4 * np.sin(8 * x)\n\n\ndef session_2_df(x):\n    return -2 * (x + 2) + 2.8 * np.cos(8 * x)\n\n\ndef session_2_ddf(x):\n    return -2 - 5.6 * np.sin(8 * x)\n\n\n# Define a general gradient ascent function that accepts the surface and derivative functions\ndef gradient_ascent(surface_func, derivative_func, start_x, learning_rate=0.1, steps=20):\n    x = start_x\n    history = [x]\n\n    for i in range(steps):\n        # Move x in the direction of the positive slope\n        x = x + learning_rate * derivative_func(x)\n        history.append(x)\n\n    return x, history\n\n\ndef plot_surface(surface_func, x_min, x_max, title, path=None):\n    x_vals = np.linspace(x_min, x_max, 400)\n    y_vals = surface_func(x_vals)\n\n    plt.figure(figsize=(10, 6))\n    plt.plot(x_vals, y_vals, label='Surface')\n    plt.title(title)\n    plt.xlabel('x')\n    plt.ylabel('f(x)')\n    plt.grid(True)\n\n    if path is not None:\n        path_x = np.array(path)\n        path_y = surface_func(path_x)\n        plt.plot(path_x, path_y, 'ro-', markersize=5, label='Gradient Ascent Path')\n        plt.plot(path_x[0], path_y[0], 'go', markersize=8, label='Start')\n        plt.plot(path_x[-1], path_y[-1], 'rx', markersize=10, label='End')\n        plt.legend()\n    plt.show()\n\n\n# Run gradient ascent from THREE different starting points.\n# This is intentional: starting location can matter on a complex surface.\nsession_2_starts = [-5, -2, 2]\nsession_2_results = []\n\nprint(\"\\n\" + \"=\" * 60)\nprint(\"PLAY SESSION 2: A BUMPY SURFACE\")\nprint(\"=\" * 60)\n\nfor start in session_2_starts:\n    peak, path = gradient_ascent(\n        session_2_surface,\n        session_2_df,\n        start_x=start,\n        learning_rate=0.08,\n        steps=80,\n    )\n\n    session_2_results.append((start, peak, path))\n\n    print(\n        f\"Start x = {start:5.1f} -> \"\n        f\"final x = {peak:8.4f}, \"\n        f\"f(x) = {session_2_surface(peak):8.4f}\"\n    )\n\n# Plot the three paths separately so we can compare how starting point\n# changes the search.\nfor start, peak, path in session_2_results:\n    plot_surface(\n        session_2_surface,\n        -6,\n        3,\n        f\"Session 2: Gradient Ascent Starting at x={start}\",\n        path,\n    )",
   "outputs": [
    {
     "kind": "text",
     "text": "\n============================================================\nPLAY SESSION 2: A BUMPY SURFACE\n============================================================\nStart x =  -5.0 -> final x =  -2.8623, f(x) =   5.3594\nStart x =  -2.0 -> final x =  -2.1464, f(x) =   6.3704\nStart x =   2.0 -> final x =  -1.4271, f(x) =   5.9493\n"
    },
    {
     "kind": "image",
     "src": "assets/img/evocomp-playground/cell-03-1.png",
     "width": 853,
     "height": 547
    },
    {
     "kind": "image",
     "src": "assets/img/evocomp-playground/cell-03-2.png",
     "width": 853,
     "height": 547
    },
    {
     "kind": "image",
     "src": "assets/img/evocomp-playground/cell-03-3.png",
     "width": 853,
     "height": 547
    }
   ]
  },
  {
   "type": "markdown",
   "source": "## Learning Rate Changes:"
  },
  {
   "type": "code",
   "source": "def session_5_surface(x):\n    return 5 - (x - 3) ** 2\n\n\ndef session_5_df(x):\n    return -2 * (x - 3)\n\n\nprint(\"\\n\" + \"=\" * 60)\nprint(\"PLAY SESSION 5: LEARNING-RATE EXPERIMENT\")\nprint(\"=\" * 60)\n\nlearning_rates = [0.05, 0.2, 0.8]\n\nfor rate in learning_rates:\n    peak, path = gradient_ascent(\n        session_5_surface,\n        session_5_df,\n        start_x=-2,\n        learning_rate=rate,\n        steps=20,\n    )\n\n    print(\n        f\"Learning rate = {rate:4.2f} -> \"\n        f\"final x = {peak:8.4f}, \"\n        f\"f(x) = {session_5_surface(peak):8.4f}\"\n    )\n\n    plot_surface(\n        session_5_surface,\n        -4,\n        6,\n        f\"Session 5: Learning Rate = {rate}\",\n        path,\n    )",
   "outputs": [
    {
     "kind": "text",
     "text": "\n============================================================\nPLAY SESSION 5: LEARNING-RATE EXPERIMENT\n============================================================\nLearning rate = 0.05 -> final x =   2.3921, f(x) =   4.6305\n"
    },
    {
     "kind": "image",
     "src": "assets/img/evocomp-playground/cell-05-1.png",
     "width": 853,
     "height": 547
    },
    {
     "kind": "text",
     "text": "Learning rate = 0.20 -> final x =   2.9998, f(x) =   5.0000\n"
    },
    {
     "kind": "image",
     "src": "assets/img/evocomp-playground/cell-05-3.png",
     "width": 853,
     "height": 547
    },
    {
     "kind": "text",
     "text": "Learning rate = 0.80 -> final x =   2.9998, f(x) =   5.0000\n"
    },
    {
     "kind": "image",
     "src": "assets/img/evocomp-playground/cell-05-5.png",
     "width": 853,
     "height": 547
    }
   ]
  },
  {
   "type": "markdown",
   "source": "## Guess and Check:"
  },
  {
   "type": "code",
   "source": "# A list of horizontal distances we want to test\nguesses = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0\n           ]\n\n# A list (vector) to store our results\nheights = []\n\nprint(\"Manual Guessing Results:\")\nprint(\"-----------------------\")\n\nfor x_guess in guesses:\n    height = f(x_guess)\n    heights.append(height)\n    print(f\"At x = {x_guess}m, the pumpkin height is {height}m\")\n\n# Use the max function to find the highest value in our vector\nmax_height = max(heights)\nprint(f\"\\nThe maximum height we found by guessing was: {max_height}m\")\n\n# Encouragement for students:\n# We stored all heights in the 'heights' list.\n# This allows us to use functions like max() to search through our data!",
   "outputs": [
    {
     "kind": "text",
     "text": "Manual Guessing Results:\n-----------------------\nAt x = 0.0m, the pumpkin height is -15.0m\nAt x = 1.0m, the pumpkin height is -6.0m\nAt x = 2.0m, the pumpkin height is 1.0m\nAt x = 3.0m, the pumpkin height is 6.0m\nAt x = 4.0m, the pumpkin height is 9.0m\nAt x = 5.0m, the pumpkin height is 10.0m\nAt x = 6.0m, the pumpkin height is 9.0m\nAt x = 7.0m, the pumpkin height is 6.0m\n\nThe maximum height we found by guessing was: 10.0m\n"
    }
   ]
  },
  {
   "type": "markdown",
   "source": "## Gradient Descent:"
  },
  {
   "type": "code",
   "source": "def df(x):\n    return -2 * (x - 5) # Corrected derivative for f(x) = 10 - (x - 5)**2\n\ndef gradient_ascent(start_x, learning_rate=0.1, steps=20):\n    x = start_x\n    history = [x]\n\n    for i in range(steps):\n        # Move x in the direction of the positive slope\n        x = x + learning_rate * df(x)\n        history.append(x)\n        if i % 5 == 0: # Print every 5 steps\n            print(f\"Step {i}: x = {x:.4f}\")\n\n    return x, history\n\n# Try it out! You can change the learning_rate to see how it affects speed.\npeak_grad, path = gradient_ascent(start_x=0.1, learning_rate=0.2)\nprint(f\"\\nGradient Ascent found peak at: {peak_grad:.4f}\")",
   "outputs": [
    {
     "kind": "text",
     "text": "Step 0: x = 2.0600\nStep 5: x = 4.7714\nStep 10: x = 4.9822\nStep 15: x = 4.9986\n\nGradient Ascent found peak at: 4.9998\n"
    }
   ]
  },
  {
   "type": "markdown",
   "source": "## Bisection Method:"
  },
  {
   "type": "code",
   "source": "def bisection_method(a, b, tolerance=0.01):\n    # We look for where the derivative df(x) is 0\n    if df(a) * df(b) > 0:\n        print(\"The peak might not be between these points!\")\n        return None\n\n    while (b - a) / 2 > tolerance:\n        midpoint = (a + b) / 2\n        if df(midpoint) == 0:\n            return midpoint\n        elif df(a) * df(midpoint) < 0:\n            b = midpoint # The peak is in the left half\n        else:\n            a = midpoint # The peak is in the right half\n\n    return (a + b) / 2\n\n# Try it out with a range from 0 to 5\npeak_bisect = bisection_method(0, 5)\nprint(f\"Bisection Method found peak near: {peak_bisect:.4f}\")",
   "outputs": [
    {
     "kind": "text",
     "text": "Bisection Method found peak near: 4.9902\n"
    }
   ]
  },
  {
   "type": "markdown",
   "source": "## Newton's Method:"
  },
  {
   "type": "code",
   "source": "def newtons_method(start_x, iterations=5):\n    x = start_x\n    print(f\"Starting Newton's Method at x = {x}\")\n\n    for i in range(iterations):\n        # Newton's update rule for optimization: x = x - f'(x) / f''(x)\n        x = x - df(x) / ddf(x)\n        print(f\"Iteration {i+1}: x = {x:.4f}, f(x) = {f(x):.4f}\")\n\n    return x\n\n# Try it out!\npeak_newton = newtons_method(start_x=0.5)\nprint(f\"\\nNewton found the peak at: {peak_newton} meters.\")",
   "outputs": [
    {
     "kind": "text",
     "text": "Starting Newton's Method at x = 0.5\nIteration 1: x = 5.0000, f(x) = 10.0000\nIteration 2: x = 5.0000, f(x) = 10.0000\nIteration 3: x = 5.0000, f(x) = 10.0000\nIteration 4: x = 5.0000, f(x) = 10.0000\nIteration 5: x = 5.0000, f(x) = 10.0000\n\nNewton found the peak at: 5.0 meters.\n"
    }
   ]
  },
  {
   "type": "markdown",
   "source": "Newton's Method was the fastest for this specific function.\n\nGradient Ascent is the most common in Machine Learning.\n\nBisection is the most reliable if you don't know the derivative.\n\nTry it yourself: Can you change the pumpkin function to  \ud835\udc53(\ud835\udc65)=10\u2212(\ud835\udc65\u22125)2  and update the derivatives to find the new peak?"
  },
  {
   "type": "markdown",
   "source": "## 2D Demo:\n# f(x,y) = (x-2)^2-xy+(y-3)^2"
  },
  {
   "type": "code",
   "source": "import numpy as np\nimport matplotlib.pyplot as plt\nfrom scipy.optimize import minimize\n\n# creating a landscape\ndef f(coordinates):\n  x, y = coordinates\n  return (x-2)**2 - x*y + (y-3)**2\n\nx_range = np.linspace(-10, 10, 100)\ny_range = np.linspace(-10, 10, 100)\nX, Y = np.meshgrid(x_range, y_range)\nZ = f([X, Y])\n\nfig = plt.figure()\nax = fig.add_subplot(111, projection='3d')\nax.plot_surface(X, Y, Z, cmap='viridis')\nax.set_xlabel('X')\nax.set_ylabel('Y')\nax.set_zlabel('Z')\nplt.show()\n\nplt.contourf(X, Y, Z, cmap='viridis')\nplt.colorbar()\nplt.xlabel('X')\nplt.ylabel('Y')\nplt.show()",
   "outputs": [
    {
     "kind": "image",
     "src": "assets/img/evocomp-playground/cell-18-0.png",
     "width": 411,
     "height": 398
    },
    {
     "kind": "image",
     "src": "assets/img/evocomp-playground/cell-18-1.png",
     "width": 570,
     "height": 438
    }
   ]
  },
  {
   "type": "code",
   "source": "x0 = np.array([0.0, 0.0])\n\nres = minimize(f, x0, method = 'nelder-mead')\nprint(res)",
   "outputs": [
    {
     "kind": "text",
     "text": "       message: Optimization terminated successfully.\n       success: True\n        status: 0\n           fun: -12.333333332846038\n             x: [ 4.667e+00  5.333e+00]\n           nit: 77\n          nfev: 151\n final_simplex: (array([[ 4.667e+00,  5.333e+00],\n                       [ 4.667e+00,  5.333e+00],\n                       [ 4.667e+00,  5.333e+00]]), array([-1.233e+01, -1.233e+01, -1.233e+01]))\n"
    }
   ]
  },
  {
   "type": "markdown",
   "source": "## Nelder-mead in Python:"
  },
  {
   "type": "code",
   "source": "import numpy as np\nfrom scipy.optimize import minimize\n\n# Define target surface function\ndef f(point):\n  x,y = point[0], point[1]\n  return (x - 2)**2 - x*y + (y - 3)**2\n\n# Initial guess (starting simplex vertex)\nx0 = np.array([0.0, 0.0])\n\n# Execute Nelder-Mead optimization\nres = minimize(f, x0, method='Nelder-Mead')\n\nprint(f\"Minimum at: {res.x}\")",
   "outputs": [
    {
     "kind": "text",
     "text": "Minimum at: [4.66664339 5.3333127 ]\n"
    }
   ]
  },
  {
   "type": "markdown",
   "source": "## Gradient Descent on 2D surface:"
  },
  {
   "type": "code",
   "source": "# Define surface gradient\ndef grad_f(v):\n    x, y = v[0], v[1]\n    dx = 2*x - y - 4\n    dy = -x + 2*y - 6\n    return np.array([dx, dy])\n# Parameters\nv = np.array([0.0, 0.0])  # Initial guess\nalpha = 0.1              # Learning rate\nepochs = 20              # Iterations\n# 2D Gradient Descent Loop\nfor epoch in range(epochs):\n    grad = grad_f(v)\n    v = v - alpha * grad\nprint(f\"Minimum at: {v}\")\n\n\n# Define surface gradient\ndef grad_f(v):\n    x, y = v[0], v[1]\n    dx = 2*x - y - 4\n    dy = -x + 2*y - 6\n    return np.array([dx, dy])\n# Parameters\nv = np.array([0.0, 0.0])  # Initial guess\nalpha = 0.1              # Learning rate\nepochs = 20              # Iterations\n# 2D Gradient Descent Loop\nfor epoch in range(epochs):\n    grad = grad_f(v)\n    v = v - alpha * grad\nprint(f'minimum at: {v}')",
   "outputs": [
    {
     "kind": "text",
     "text": "Minimum at: [4.05904937 4.72518409]\nminimum at: [4.05904937 4.72518409]\n"
    }
   ]
  },
  {
   "type": "markdown",
   "source": "## Newton's method:"
  },
  {
   "type": "code",
   "source": " #Define Gradient & Hessian\n\ndef grad(xy):\n  x, y = xy[0], xy[1]\n  return np.array([2*(x-2) - y, -x + 2*(y-3)])\n\ndef hessian(xy):\n  return np.array([[2.0, -1.0], [-1.0, 2.0]])\n\n# Iteration Step\nxy = np.array([0.0, 0.0])\n\nfor _ in range(5):\n  g = grad(xy)\n  h  = hessian(xy)\nxy = xy - np.linalg.solve(h, g)\n\nprint('final point:',xy)\nprint('minimum:',f(xy))\n\nplt.contourf(X, Y, Z, cmap='viridis')\nplt.colorbar()\nplt.xlabel('X')\nplt.ylabel('Y')\nplt.plot(xy[0], xy[1], 'ro')\nplt.show()",
   "outputs": [
    {
     "kind": "text",
     "text": "final point: [4.66666667 5.33333333]\nminimum: 1013.691358024691\n"
    },
    {
     "kind": "image",
     "src": "assets/img/evocomp-playground/cell-25-1.png",
     "width": 566,
     "height": 438
    }
   ]
  },
  {
   "type": "markdown",
   "source": "## Nelder-Mead:"
  },
  {
   "type": "code",
   "source": "import numpy as np\nimport matplotlib.pyplot as plt\nfrom scipy.optimize import minimize\n\n# Define the terrain function\ndef f(coords):\n    x, y = coords\n    return (x**2 - 2)**2 - x*y + (y**2 - 3)**2 #X vs X**2\n\n# Visualize the terrain\nx_range = np.linspace(-5, 10, 100)\ny_range = np.linspace(-5, 10, 100)\nX, Y = np.meshgrid(x_range, y_range)\nZ = f([X, Y])\n\n# Plot the figure\nplt.figure(figsize=(8, 6))\nplt.contourf(X, Y, Z, levels=50, cmap='terrain')\nplt.colorbar(label='Height (m)')\nplt.title('Surveyor\\'s Map of the Terrain')\nplt.xlabel('East (x)')\nplt.ylabel('North (y)')\nplt.show()",
   "outputs": [
    {
     "kind": "image",
     "src": "assets/img/evocomp-playground/cell-28-0.png",
     "width": 693,
     "height": 547
    }
   ]
  },
  {
   "type": "code",
   "source": "# Start at an arbitrary guess: (0, 0)\ninitial_guess = [0, 0]\n\n# Nelder-Mead is a 'Direct Search' method\nresult_nm = minimize(f, initial_guess, method='Nelder-Mead')\n\nprint(f\"Nelder-Mead found the minimum at x = {result_nm.x[0]:.2f}, y = {result_nm.x[1]:.2f}\")\nprint(f\"The height at this point is {result_nm.fun:.2f} meters.\")",
   "outputs": [
    {
     "kind": "text",
     "text": "Nelder-Mead found the minimum at x = 1.52, y = 1.79\nThe height at this point is -2.58 meters.\n"
    }
   ]
  },
  {
   "type": "markdown",
   "source": "## Gradient Descent:"
  },
  {
   "type": "code",
   "source": "import numpy as np\n\n# In the Gradient Descent cell, change 0.1 to 0.8 or 0.001. Does it still find the bottom?\n# New Terrain: Can you change the f(coords) function to x**2 + y**2 and see if the code still works?\n\ndef gradient(coords):\n    x, y = coords\n    # These formulas represent how steep the hill is in the x and y directions\n    df_dx = 2*(x - 2) - y\n    df_dy = -x + 2*(y - 3)\n    return np.array([df_dx, df_dy])\n\n# Simple Gradient Descent Loop\nposition = np.array([0.0, 0.0]) # Starting point\nlearning_rate = .1             # How big each step is\n\nfor i in range(50):\n    grad = gradient(position)\n    position = position - learning_rate * grad # Step away from the 'uphill' direction\n\nprint(f\"Gradient Descent minimum at x = {position[0]:.2f}, y = {position[1]:.2f}\")",
   "outputs": [
    {
     "kind": "text",
     "text": "Gradient Descent minimum at x = 4.64, y = 5.31\n"
    }
   ]
  },
  {
   "type": "code",
   "source": "import numpy as np\nimport matplotlib.pyplot as plt\nfrom matplotlib import cm\n\n# ---------------------------------------------------------\n# Define the terrain function\n# ---------------------------------------------------------\ndef f(coords):\n    x, y = coords\n    return (x**2 - 2)**2 - x*y + (y**2 - 3)**2\n\n# ---------------------------------------------------------\n# Custom Nelder-Mead that logs every simplex (triangle) step\n# ---------------------------------------------------------\ndef nelder_mead(f, x0, step=1.0, tol=1e-6, max_iter=60,\n                 alpha=1.0, gamma=2.0, rho=0.5, sigma=0.5):\n    dim = len(x0)\n    x0 = np.array(x0, dtype=float)\n\n    # Build the initial simplex (dim+1 points for a dim-D problem)\n    simplex = [x0]\n    for i in range(dim):\n        pt = x0.copy()\n        pt[i] += step\n        simplex.append(pt)\n    simplex = np.array(simplex)\n\n    history = [simplex.copy()]  # store every simplex for plotting\n\n    for it in range(max_iter):\n        # 1. Order points by function value (best to worst)\n        scores = np.array([f(p) for p in simplex])\n        order = np.argsort(scores)\n        simplex = simplex[order]\n        scores = scores[order]\n\n        best, worst, second_worst = scores[0], scores[-1], scores[-2]\n\n        # Convergence check\n        if np.std(scores) < tol:\n            break\n\n        # 2. Centroid of all points except the worst\n        centroid = simplex[:-1].mean(axis=0)\n\n        # 3. Reflection\n        x_r = centroid + alpha * (centroid - simplex[-1])\n        f_r = f(x_r)\n\n        if scores[0] <= f_r < second_worst:\n            simplex[-1] = x_r\n\n        elif f_r < best:\n            # 4. Expansion\n            x_e = centroid + gamma * (x_r - centroid)\n            f_e = f(x_e)\n            simplex[-1] = x_e if f_e < f_r else x_r\n\n        else:\n            # 5. Contraction\n            x_c = centroid + rho * (simplex[-1] - centroid)\n            f_c = f(x_c)\n            if f_c < worst:\n                simplex[-1] = x_c\n            else:\n                # 6. Shrink toward best point\n                simplex = simplex[0] + sigma * (simplex - simplex[0])\n\n        history.append(simplex.copy())\n\n    return simplex[np.argmin([f(p) for p in simplex])], history\n\n\n# Run the optimizer, starting far from the minimum\nx0 = [-4, 8]\nbest_point, history = nelder_mead(f, x0, step=1.5)\nprint(f\"Optimal point found: {best_point}, f = {f(best_point):.4f}\")\nprint(f\"Converged in {len(history)} simplex steps\")\n\n\n# Visualize the terrain + every simplex step\nx_range = np.linspace(-10, 10, 200)\ny_range = np.linspace(-10, 10, 200)\nX, Y = np.meshgrid(x_range, y_range)\nZ = f([X, Y])\n\nplt.figure(figsize=(9, 7))\nplt.contourf(X, Y, Z, levels=50, cmap='terrain')\nplt.colorbar(label='Height (m)')\n\nn_steps = len(history)\ncolors = cm.cool(np.linspace(0, 1, n_steps))  # blue = early, pink = late\n\nfor i, simplex in enumerate(history):\n    triangle = np.vstack([simplex, simplex[0]])  # close the triangle\n    plt.plot(triangle[:, 0], triangle[:, 1],\n              color=colors[i], linewidth=1.2, alpha=0.8, marker='o', markersize=3)\n\n# Mark start and final optimum\nplt.plot(*x0, 'ks', markersize=10, label='Start')\nplt.plot(*best_point, 'r*', markersize=18, label='Optimum found')\n\nplt.title(\"Nelder-Mead Simplex tracked\")\nplt.xlabel('East (x)')\nplt.ylabel('North (y)')\nplt.legend()\nplt.show()",
   "outputs": [
    {
     "kind": "text",
     "text": "Optimal point found: [1.51492308 1.7919822 ], f = -2.5831\nConverged in 49 simplex steps\n"
    },
    {
     "kind": "image",
     "src": "assets/img/evocomp-playground/cell-32-1.png",
     "width": 784,
     "height": 624
    }
   ]
  },
  {
   "type": "code",
   "source": "import numpy as np\nimport matplotlib.pyplot as plt\nfrom matplotlib import cm\n\n\n# Define the terrain function\ndef f(coords):\n    x, y = coords\n    return (x**2 - 2)**2 - x*y + (y**2 - 3)**2\n\n\n# Custom Nelder-Mead that logs every simplex step\n\ndef nelder_mead(f, x0, step=1.0, tol=1e-6, max_iter=60,\n                 alpha=1.0, gamma=2.0, rho=0.5, sigma=0.5):\n    dim = len(x0)\n    x0 = np.array(x0, dtype=float)\n\n    simplex = [x0]\n    for i in range(dim):\n        pt = x0.copy()\n        pt[i] += step\n        simplex.append(pt)\n    simplex = np.array(simplex)\n\n    history = [simplex.copy()]\n\n    for it in range(max_iter):\n        scores = np.array([f(p) for p in simplex])\n        order = np.argsort(scores)\n        simplex = simplex[order]\n        scores = scores[order]\n\n        best, worst, second_worst = scores[0], scores[-1], scores[-2]\n\n        if np.std(scores) < tol:\n            break\n\n        centroid = simplex[:-1].mean(axis=0)\n        x_r = centroid + alpha * (centroid - simplex[-1])\n        f_r = f(x_r)\n\n        if scores[0] <= f_r < second_worst:\n            simplex[-1] = x_r\n        elif f_r < best:\n            x_e = centroid + gamma * (x_r - centroid)\n            f_e = f(x_e)\n            simplex[-1] = x_e if f_e < f_r else x_r\n        else:\n            x_c = centroid + rho * (simplex[-1] - centroid)\n            f_c = f(x_c)\n            if f_c < worst:\n                simplex[-1] = x_c\n            else:\n                simplex = simplex[0] + sigma * (simplex - simplex[0])\n\n        history.append(simplex.copy())\n\n    return simplex[np.argmin([f(p) for p in simplex])], history\n\n\n# Run TWO searches: original start, and its mirror (-x, -y)\nx0_a = [-4, 8]\nx0_b = [4, -8]   # mirrored starting point -> pulls toward the OTHER minimum\n\nbest_a, history_a = nelder_mead(f, x0_a, step=1.5)\nbest_b, history_b = nelder_mead(f, x0_b, step=1.5)\n\nprint(f\"Run A: start {x0_a} -> optimum {best_a}, f = {f(best_a):.4f}\")\nprint(f\"Run B: start {x0_b} -> optimum {best_b}, f = {f(best_b):.4f}\")\n\n\n# Visualize the terrain + both simplex walks\nx_range = np.linspace(-10, 10, 200)\ny_range = np.linspace(-10, 10, 200)\nX, Y = np.meshgrid(x_range, y_range)\nZ = f([X, Y])\n\nplt.figure(figsize=(9, 7))\nplt.contourf(X, Y, Z, levels=50, cmap='terrain')\nplt.colorbar(label='Height (m)')\n\ndef plot_history(history, cmap):\n    n_steps = len(history)\n    colors = cmap(np.linspace(0, 1, n_steps))\n    for i, simplex in enumerate(history):\n        triangle = np.vstack([simplex, simplex[0]])\n        plt.plot(triangle[:, 0], triangle[:, 1],\n                  color=colors[i], linewidth=1.0, alpha=0.8, marker='o', markersize=2)\n\nplot_history(history_a, cm.cool)\nplot_history(history_b, cm.autumn)\n\nplt.plot(*x0_a, 'ks', markersize=10, label='Start A')\nplt.plot(*best_a, 'b*', markersize=18, label='Optimum A')\nplt.plot(*x0_b, 'k^', markersize=10, label='Start B')\nplt.plot(*best_b, 'r*', markersize=18, label='Optimum B (other spot)')\n\nplt.title(\"Nelder-Mead: Two Starting Points Finding different Optima\")\nplt.xlabel('East (x)')\nplt.ylabel('North (y)')\nplt.legend()\nplt.show()",
   "outputs": [
    {
     "kind": "text",
     "text": "Run A: start [-4, 8] -> optimum [1.51492308 1.7919822 ], f = -2.5831\nRun B: start [4, -8] -> optimum [-1.51570265 -1.79235995], f = -2.5831\n"
    },
    {
     "kind": "image",
     "src": "assets/img/evocomp-playground/cell-33-1.png",
     "width": 784,
     "height": 624
    }
   ]
  }
 ]
};
