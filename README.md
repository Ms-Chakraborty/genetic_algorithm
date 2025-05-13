🏥 Nurse Scheduling Optimization using Genetic Algorithm
This project uses a Genetic Algorithm (GA) to optimize weekly nurse schedules based on constraints like maximum working hours per day/week, using real-world data loaded from Excel files. The goal is to evenly distribute work among nurses while respecting scheduling constraints.

**📋 Features**
Load nurse data from Excel spreadsheets.

Generate initial schedules respecting hour limits.

Evolve schedules over generations to improve fairness.

Apply crossover and mutation to simulate natural selection.

Identify and print distinct optimal schedules.

Handles specialization categories (e.g., Pediatric, Trauma, etc.).

**🧠 Optimization Objectives**
Fairness: Minimize variance in hours assigned per nurse.

Constraints:

Max 8 hours/day per nurse.
.**📁 File Structure**
-  **nurse_scheduler.py      # Main Python script**
-  **nurse_data_original.xlsx  # Original immutable nurse dataset**
-  **nurse_data_temp.xlsx      # Modifiable copy of nurse data**
-  **README.md               # Project documentation**

- **  📦 Requirements**
Python 3.8+

Packages:

numpy

pandas

openpyxl (for reading/writing Excel files)


Max 40 hours/week per nurse.

- **Penalty System:**
Penalizes violations of the above constraints.

Fitness is negatively impacted by overworking any nurse.
- 🧬 **Algorithm Parameters**
You can adjust these at the top of nurse_scheduler.py:
D = 7           # Number of days
H = 8           # Hours per day
POP_SIZE = 20   # Population size
GENERATIONS = 1000
CROSSOVER_RATE = 0.7
MUTATION_RATE = 0.01
- **📝 Example Output**
Original Data:
  Nurse Name  Specialization
0    Alice    Pediatric Nurse
1    Bob      General
...

Best Schedule:
Nurse 1:
  Day 1: 11100000
  ...
Nurse 2:
  Day 1: 00011100
  ...
