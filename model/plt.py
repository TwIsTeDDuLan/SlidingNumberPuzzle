import matplotlib.pyplot as plt



x = [20,40,20,30,10,10,20,20,20,30]
y = [30,60,40,60,30,40,40,50,30,70]

plt.scatter(x,y, label="soda", color="red", marker="o")

plt.title("Simple Line Plot")
plt.xlabel("X-axis Label")
plt.ylabel("Y-axis Label")

plt.show()