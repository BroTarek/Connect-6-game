**Capital University** [formerly **Helwan University**] | the **Faculty of Computing & Artificial Intelligence**  
the **Mainstream Program** [both the **Computer Science**/**Artificial Intelligence** Depts.] &, the **Software Engineering Program**  
**Module:** AI310 & CS361 Artificial Intelligence – Fall "Semester 1" 2026-2027

---

# AI310/CS361 Artificial Intelligence – Fall 2026
## Course Project Instructions

### Instructions to Students

- This is a group work project. Each group consists of **six students** ("Mainstream Programme") or **four to six** ("Software Engineering Programme"). *The Teaching Assistant must approve group members through registration.* Each group must develop the assigned idea using Python.
  - **Project Objectives:** The objectives of this project can be summarised as applying the main ideas, fundamental concepts, and basic algorithms in the fields of artificial intelligence and machine learning.
- **Submission:** Submission is done according to the following schedule:
  - **Week 12/13: Submission and Discussion of the (1) Project and (2) Documentation.** *The report should include the following: (1) Project idea in detail, (2) Main functionalities, (3) Similar applications in the market, (4) A literature review of Academic publications (papers) relevant to the idea (at least 4 to 6 papers, as per the number of team members), (5) the Dataset employed (preferably a publicly available dataset), (6) Details of the algorithm(s)/approach(es) used and the results of the experiments, and (7) Development platform.*
- **Assessment:** Assessment will be based on the reports, code submitted, and discussions with team members. All team members must contribute across all phases, and each report must clearly state each member's role.
  - The Project will be assessed based on the following criteria:
    - The complexity of the problem & the correctness of the algorithms employed.
    - The quality/comprehensiveness of your experiments & documentation.
    - The correctness of your analysis and design diagrams.
    - Implementation correctness.
- **Feedback:** If requested, further details and feedback could be provided for each group through discussions with the teaching assistant(s) during the weekly-labs/office-hours.
- You can only submit your own work. Any student suspected of plagiarism will be subject to the procedures set out by the Faculty/University (including failing the course entirely).
  - **Academic Integrity:** The University's policies on academic integrity will be enforced against students who violate the University's standards of academic integrity. Examples of behaviour that are not allowed are:
    - Copying all or part of someone else's work and submitting it as your own;
    - Giving another student in the class a copy of your work; and
    - Copying parts from the internet, textbooks, etc.
    - If you have any questions concerning what is allowed, please don't hesitate to discuss them with me.
- **We understand that you might positively refer to AI tools to support your research. If there are any concerns about AI-generated submission content, you will be invited to verify and defend your work.**

---

## 2) An Intelligent Connect-6 Player using the Minimax Algorithm, Alpha-Beta Pruning, and Heuristic Functions

**Description:** The project aims to develop an intelligent player for Connect-6, an extended version of Connect Four in which players strive to connect six markers in a row horizontally, vertically, or diagonally on a 19x19 grid. This board game introduces increased complexity compared to Connect Four, making it a challenging problem for artificial intelligence (AI) systems to navigate strategically.

### Project Requirements

- **Minimax Algorithm Implementation:** The project requires implementing the Minimax algorithm, a decision-making approach commonly used in two-player games. Minimax explores potential moves by recursively evaluating game states to determine the optimal strategy for maximising wins or minimising losses.
- **Alpha-Beta Pruning Integration:** Integrating Alpha-Beta Pruning with the Minimax algorithm is essential to optimise the AI player's decision-making. Alpha-Beta Pruning reduces the number of evaluated game states, making the algorithm more efficient without sacrificing strategic depth.
- **Heuristic Function Design:** Develop heuristic functions that quickly evaluate a given board state. Heuristics should capture patterns and configurations conducive to winning, guiding the AI player's decisions when exhaustive search is impractical.