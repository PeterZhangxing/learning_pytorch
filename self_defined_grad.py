import numpy as np

def compute_error_for_line_given_points(b,w,points):
    total_erro = 0
    for i in range(len(points)):
        x = points[i,0]
        y = points[i,1]
        loss = (y-(w*x+b))**2
        total_erro += loss
    return total_erro / float(len(points))

def step_gradient(b_current, w_current, points, learningRate):
    b_gradient = 0
    w_gradient = 0
    N = float(len(points))
    for i in range(len(points)):
        x = points[i,0]
        y = points[i,1]
        b_gradient += -(2/N) * (y - ((w_current * x) + b_current))
        w_gradient += -(2/N) * x * (y - ((w_current * x) + b_current))
    new_b = b_current - learningRate * b_gradient
    new_w = w_current - learningRate * w_gradient
    return new_b,new_w

def gradient_descent_runner(points, starting_b, starting_w, learning_rate, num_iterations):
    b = starting_b
    w = starting_w
    for i in range(num_iterations):
        b,w = step_gradient(b,w,points,learning_rate)
    return b,w

def run():
    points = np.genfromtxt("data/test/data.csv", delimiter=",")
    learning_rate = 0.0001
    initial_b = 0 # initial y-intercept guess
    initial_w = 0 # initial slope guess
    num_iterations = 1000

    init_erro = compute_error_for_line_given_points(initial_b, initial_w, points)
    print("Starting gradient descent at b = {}, m = {}, error = {}".format(initial_b, initial_w,init_erro))

    b,w = gradient_descent_runner(points, initial_b, initial_w, learning_rate, num_iterations)
    final_erro = compute_error_for_line_given_points(b, w, points)
    print("After {} iterations b = {}, m = {}, error = {}".format(num_iterations, b, w,final_erro))

    return None

if __name__ == '__main__':
    run()