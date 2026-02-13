
plt.figure(figsize=(10,5))
plt.hist(error,bins=10)

plt.title("Prediction Error DIstribution")
plt.xlabel("Error")
plt.ylabel("Frequency")
plt.show()