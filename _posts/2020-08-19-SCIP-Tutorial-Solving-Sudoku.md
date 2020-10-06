---
layout: post
title:  "SCIP Tutorial: How to solve Sudoku (C++)"
date: 2020-08-19
categories: Integer-Programming
use_math: true
description: In this post, we will discuss how to solve a Sudoku Puzzle using SCIP and programmed in c++.
header-includes:
---

I assume the reader already has familiarity with the Sudoku puzzle. If not, go here: [Sudoku Wikipedia](https://en.wikipedia.org/wiki/Sudoku). I also assume you have SCIP optimization suite installed and if not, see the instructions here: [Getting Started with SCIP optimization in C++: A toy example](https://www.cgudapati.com/integer-programming/2019/12/15/Getting-Started-With-SCIP-Optimization-Suite.html)

Let us start with the directory structure. We create an empty directory named `sudoku_scip\`. Inside we create another empty directory named `src\`. Inside `sudoku` directory, we also have a `Makefile` and a `data` directory with list of puzzles. 

```bash
$ ls sudoku
Makefile  README.md data      src
$ ls sudoku/src
sudoku_main.cpp sudoku_utils.h
```

Let us start looking at the non-scip related file, the `sudoku_utils.h`

```c++

#include <iostream>
#include <fstream>
#include <vector>
#include <string>

//Here we create a simple namespace to hold all of the related suDoKu functions together.
namespace sudoku
{
    std::vector<std::vector<int>> get_sudoku_grid(std::string file_path)
    {
        //Just setting an 9x9 grid for storing the sudoku puzzle.
        std::vector<std::vector<int>> puzzle(9, std::vector<int>(9));

        //Let us read the puzzle into a stringstream
        std::ifstream infile(file_path);

        std::string puzzle_data;

        if (infile.is_open())
        {
        std::getline(infile, puzzle_data);
            if (puzzle_data.length() != 81)      //The puzzle should have 81 characters
            {
                std::cerr << "Please check the puzzle file for inconsistencies"
                          << "\n";
                exit(1);
            }
        }

        int idx = 0; //This variable will be used to access the numbers in the puzzle string

        for (int i = 0; i < 9; ++i)
        {
            for (int j = 0; j < 9; ++j)
            {
                if ((puzzle_data.substr(idx, 1) != ".") and (puzzle_data.substr(idx, 1) != "0"))  // We will only convert the numeric charcater to an integer if it is not '.' or '0'.
                {
                    puzzle[i][j] = std::stoi(puzzle_data.substr(idx, 1));
                }
                else
                {
                    puzzle[i][j] = -1;    // If we are currently reading a '.' or '0' make it -1.
                }
                idx++;
            }
        }

        return puzzle;
    }

    void print_sudoku(const std::vector<std::vector<int>> &sudoku_puzzle)
    {
        std::cout << "+----------+-----------+-----------+"
                  << "\n";
        for (auto i = 0; i < 9; ++i)
        {
            std::cout << "|";
            for (auto j = 0; j < 9; ++j)
            {
                if (sudoku_puzzle[i][j] > 0)
                {

                    if (j == 2 or j == 5 or j == 8)
                    {
                        std::cout << sudoku_puzzle[i][j] << " | ";
                    }
                    else
                    {
                        std::cout << sudoku_puzzle[i][j] << "   ";
                    }
                }
                else
                {
                    if (j == 2 or j == 5 or j == 8)
                    {
                        std::cout << "."
                                  << " | ";
                    }
                    else
                    {
                        std::cout << "."
                                  << "   ";
                    }
                }
            }
            std::cout << "\n";

            if (i == 2 or i == 5 or i == 8)
            {
                std::cout << "+----------+-----------+-----------+"
                          << "\n";
            }
        }
    }
} // namespace sudoku
```



The `sudoku_utils.h` file is commented well enough so that the reader can understand it fairly easily. The `get_sudoku_grid()` function is used to read the sudoku puzzle from a file which has a single line with 81 characters. the numbers 1,2...9 are represented as is in the file and the blanks can either be a '.' or a '0' character. 

The next function just displays the Sudoku grid. This function is called when we first read the puzzle and then when we solve the puzzle

```c++
     1	    if (args < 2)
     2	    {
     3	        std::cerr << "call " << argv[0] << " <puzzle file> " << "\n";
     4	        exit(1);
     5	    }
     6	
     7	    std::string puzzle_file_path = argv[1];
     8	
     9	    auto puzzle = sudoku::get_sudoku_grid(puzzle_file_path);
    10	
    11	    std::cout << "The unsolved Sudoku Puzzle is: " << "\n";
    12	    sudoku::print_sudoku(puzzle);
    13	
    14	    //Setting up the SCIP environment
    15	    
    16	    SCIP *scip = nullptr; //Declaring the scip environment
    17	
    18	    SCIP_CALL(SCIPcreate(&scip)); //Creating the SCIP environment
    19	
    20	    /* include default plugins */
    21	    SCIP_CALL(SCIPincludeDefaultPlugins(scip));
    22	
    23	    //Creating the SCIP Problem.
    24	    SCIP_CALL(SCIPcreateProbBasic(scip, "SUDOKU"));
    25	
    26	    SCIP_CALL(SCIPsetObjsense(scip, SCIP_OBJSENSE_MINIMIZE)); //Does not matter for us as this is just a feasibility problem
```

 

 Lines 1-5 just tell us how to call SCIP accurately. Line 9 loads the sudoku puzzle read from the disk into the `puzzle` variable.  Line 11-12 print the unsolved puzzle. 

Line 14 - 25 shows us how to set the scip environment.  Line 26  lets us set the obj-sense but since sudoku is a feasibility problem, it doesn't matter that much

Let us take a look at the integer programming model for Sudoku
$$
\begin{alignat}{2}
& &   \sum_{i=1}^{9}& x_{ijk}&= 1 \quad \text{for} \quad j,k = 1, \dots 9\label{eq:constraint1}\\
& &   \sum_{j=1}^{9}& x_{ijk}&= 1\quad \text{for} \quad i,k = 1, \dots 9\label{eq:constraint2}\\
& \sum_{j=3p-2}^{3p}& \sum_{i=3q-2}^{3q}& x_{ijk}&= 1\quad \text{for} \quad k = 1, \dots 9 \ \text{and}\ p,q = 1,2,3  \ \label{eq:constraint3}\\
& &   \sum_{k=1}^{9}& x_{ijk}&= 1 \quad \text{for} \quad i,j = 1, \dots 9 \label{eq:constraint4}\\
& &   & x_{ijk}&= 1\quad \text{for}\quad (i,j,k) \in \text{known cells}  \label{eq:constraint5}
\end{alignat}
$$
The standard Sudoku grid has 9x9 = 81 squares. Some of them will already be filled with numbers given in the puzzle and we should fill the blanks. So, each square in the grid can take 9 numbers. Let $x_{ijk}$  represent all the binary decision variables. if $ x_{ijk} = 1 $, then the number k is present in ith row and jth column i.e if $x_{1,4,7} = 1$, then the number 7 is present in 1st row and 4th column.

 We will create the binary sudoku variables in the next code section

```c++
     1	std::vector<std::vector<std::vector<SCIP_VAR *>>> x_vars(9, std::vector<std::vector<SCIP_VAR *>>(9, std::vector<SCIP_VAR *>(9)));
     2	
     3	    std::ostringstream namebuf;
     4	
     5	    for (int i = 0; i < 9; ++i)
     6	    {
     7	        for (int j = 0; j < 9; ++j)
     8	        {
     9	            for (int k = 0; k < 9; ++k)
    10	            {
    11	                SCIP_VAR *var = nullptr;
    12	                namebuf.str("");
    13	                namebuf << "x[" << i << "," << j << "," << k << "]";
    14	                SCIP_CALL(SCIPcreateVarBasic(scip,                  // SCIP environment
    15	                                             &var,                  // reference to the variable
    16	                                             namebuf.str().c_str(), // name of the variable
    17	                                             0.0,                   // Lower bound of the variable
    18	                                             1.0,                   // upper bound of the variable
    19	                                             1.0,                   // Obj. coefficient. Doesn't really matter for this problem
    20	                                             SCIP_VARTYPE_BINARY    // Binary variable
    21	                                             ));
    22	                SCIP_CALL(SCIPaddVar(scip, var));
    23	                x_vars[i][j][k] = var;
    24	            }
    25	        }
    26	    }
```

We are using three for loops to create the variables. Since these variables are not going to be modified in anyway, we can use the simple  `SCIPcreateVarBasic` command. 

After creating the variables, let us create the constraints. Our first constraint ensures that in each column, the numbers 1, 2, ..., 9 do not repeat

```c++

    std::vector<SCIP_CONS *> column_constrs;   //These constraints will model that fact that in each column, the numbers 1..9 should not repeat. 

    for (int j = 0; j < 9; ++j)
    {
        for (int k = 0; k < 9; ++k)
        {
            SCIP_CONS *cons = nullptr;

            namebuf.str("");
            namebuf << "col_" << j << "_" << k << "]";
            SCIP_CALL(SCIPcreateConsBasicLinear(scip,
                                                &cons,
                                                namebuf.str().c_str(),
                                                0,
                                                nullptr,
                                                nullptr,
                                                1.0,
                                                1.0));
            for (int i = 0; i < 9; ++i)            // The constraint will look like x_1jk + x_2jk + x_3jk + ... + x_9jk = 1 for a given value of j and k
            {
                SCIP_CALL(SCIPaddCoefLinear(scip, cons, x_vars[i][j][k], 1.0));
            }

            SCIP_CALL(SCIPaddCons(scip, cons));
            column_constrs.push_back(cons);
        }
    }
```

After that we will add another constraint that will ensure that in each row, the numbers 1, 2, ..., 9 do not repeat

```c++
std::vector<SCIP_CONS *> row_constrs;  //These constraints will model that fact that in each row, the numbers 1..9 do not repeat. 

    for (int i = 0; i < 9; ++i)
    {
        for (int k = 0; k < 9; ++k)
        {
            SCIP_CONS *cons = nullptr;

            namebuf.str("");
            namebuf << "row_" << i << "_" << k << "]";
            SCIP_CALL(SCIPcreateConsBasicLinear(scip,
                                                &cons,
                                                namebuf.str().c_str(),
                                                0,
                                                nullptr,
                                                nullptr,
                                                1.0,
                                                1.0));
            for (int j = 0; j < 9; ++j)          // The constraint will look like x_i1k + x_i2k + x_i3k + ... + x_i9k = 1 for a given value of i and k
            {
                SCIP_CALL(SCIPaddCoefLinear(scip, cons, x_vars[i][j][k], 1.0));
            }

            SCIP_CALL(SCIPaddCons(scip, cons));
            row_constrs.push_back(cons);
        }
    }
```



Now in a traditional sudoku puzzle, each sub-grid should contain the numbers 1, 2, ..., 9 without repeating. We have the following constraints

```c++
//Subgrid constraints
std::vector<SCIP_CONS *> subgrid_constrs;  // These constraints will model that each of the 3x3 subgrids will contain 1...9 without any repetition. 

for (int k = 0; k < 9; ++k)
{
    for (int p = 0; p < 3; ++p)
    {
        for (int q = 0; q < 3; ++q)
        {
            SCIP_CONS *cons = nullptr;

            namebuf.str("");
            namebuf << "subgrid_" << k << "_" << p << "_" << q << "]";
            SCIP_CALL(SCIPcreateConsBasicLinear(scip,
                                                &cons,
                                                namebuf.str().c_str(),
                                                0,
                                                nullptr,
                                                nullptr,
                                                1.0,
                                                1.0));
            for (int j = 3 * (p + 1) - 3; j < 3 * (p + 1); ++j)                    //since we are indexing from 0..8 be careful with the loop indices. 
            {
                for (int i = 3 * (q + 1) - 3; i < 3 * (q + 1); ++i)
                {
                    SCIP_CALL(SCIPaddCoefLinear(scip, cons, x_vars[i][j][k], 1.0));
                }
            }
            SCIP_CALL(SCIPaddCons(scip, cons));
            subgrid_constrs.push_back(cons);
        }
    }
}
```
If it is confusing, try to write a constraint by hand for the values  p = 0, q = 0 and k =1...9 and see how it looks. We are just writing the constraints from the math notation to code. 

The next set of constraints ensure that we should fill every position in the 9x9 grid with one number.

And finally we have to assign the numbers that are already given in the initial puzzle to the corresponding variables. We can do this using the `SCIPfixVar()` function. The `infeasible` and `fixed` variables can be used for debugging.



```c++
  //We have to set assign the already given numbers to the corresponding variables and we use the SCIPfixVar() function

    SCIP_Bool infeasible;
    SCIP_Bool fixed;
    for (int i = 0; i < 9; ++i)
    {
        for (int j = 0; j < 9; ++j)
        {
            if (puzzle[i][j] > 0)
            {
                SCIP_CALL(SCIPfixVar(scip, x_vars[i][j][(puzzle[i][j]) - 1], 1.0, &infeasible, &fixed));
            }
        }
    }

```

And now we solve the problem by specifying an objective sense. This is just to demonstrate how to set objective sense (minimize of maximize). Since this is a feasibility problem, it doesn't matter whether the obj. sense is minimization or maximization. We also show how to turn off the logging by setting an int parameter. And finally, we solve the problem using the `SCIPsolve()` function and store the solution status afterwards. 

    SCIP_CALL(SCIPsetObjsense(scip, SCIP_OBJSENSE_MAXIMIZE));
    
    SCIP_CALL(SCIPsetIntParam(scip, "display/verblevel", 0));   // We use SCIPsetIntParams to turn off the logging. 
    
    SCIP_CALL(SCIPsolve(scip));
    
    SCIP_STATUS soln_status = SCIPgetStatus(scip);   // Some wrongly generated sudoku puzzles can be infeasible. So we use the solnstatus to display different types of output.
And finally, we print the solution grid and free the variables.



```c++
    if (soln_status == 11)                           // solution status of 11 indicates optimal solution was found. Hence we can print the final puzzle.
    {
        SCIP_SOL *sol;
        sol = SCIPgetBestSol(scip);

        for (int i = 0; i < 9; ++i)
        {
            for (int j = 0; j < 9; ++j)
            {
                for (int k = 0; k < 9; ++k)
                {
                    if (SCIPgetSolVal(scip, sol, x_vars[i][j][k]) > 0)
                    {
                        puzzle[i][j] = k + 1;     // We are using 0  based indices. so when we have to display the final puzzle, we should increment it by 1.
                    }
                }
            }
        }
        std::cout << "The solved puzzle is: " << "\n";
        sudoku::print_sudoku(puzzle);
    }
    else if( soln_status == 12)                    // solutions status of 12 indicates that the puzzle is infeasible. 
    {
        std::cout << "Check the Input puzzle" << "\n";
    }

    //Freeing the variables

    for (int i = 0; i < 9; ++i)
    {
        for (int j = 0; j < 9; ++j)
        {
            for (int k = 0; k < 9; ++k)
            {
                SCIP_CALL(SCIPreleaseVar(scip, &x_vars[i][j][k]));
            }
        }
    }
    x_vars.clear();

    // Freeing the constraints

    for (auto &constr : column_constrs)
    {
        SCIP_CALL(SCIPreleaseCons(scip, &constr));
    }
    column_constrs.clear();

    for (auto &constr : row_constrs)
    {
        SCIP_CALL(SCIPreleaseCons(scip, &constr));
    }
    row_constrs.clear();

    for (auto &constr : subgrid_constrs)
    {
        SCIP_CALL(SCIPreleaseCons(scip, &constr));
    }
    subgrid_constrs.clear();

    for (auto &constr : fillgrid_constrs)
    {
        SCIP_CALL(SCIPreleaseCons(scip, &constr));
    }
    fillgrid_constrs.clear();

    //freeing scip

    SCIP_CALL(SCIPfree(&scip));
}

```

It is always useful to see the LP to glean more information about how the model is being constructed. To write the model to an LP we can use the following code

```c++
SCIP_CALL((SCIPwriteOrigProblem(scip, "scip_sudoku.lp", nullptr, FALSE)));
```

And to run this program on a unix-based system follow the following commands

```bash
Sudoku$ ls
Makefile  README.md bin       data      obj       src

Sudoku$ make
-> compiling obj/static/O.darwin.x86_64.gnu.opt/sudoku_main.o
g++ -Isrc -DWITH_SCIPDEF -DFNAME_LCASE_DECOR -I/Library/scip7.0.1/scip/src -DNDEBUG -DSCIP_ROUNDING_FE  -DNO_CONFIG_HEADER -DNPARASCIP -DSCIP_WITH_ZLIB  -DSCIP_WITH_GMP  -DSCIP_WITH_READLINE   -O3 -fomit-frame-pointer -mtune=native     -std=c++0x -fno-stack-check -pedantic -Wno-long-long -Wall -W -Wpointer-arith -Wcast-align -Wwrite-strings -Wshadow -Wno-unknown-pragmas -Wno-unused-parameter -Wredundant-decls -Wdisabled-optimization -Wctor-dtor-privacy -Wnon-virtual-dtor -Wreorder -Woverloaded-virtual -Wsign-promo -Wsynth -Wcast-qual -Wno-unused-parameter -Wno-strict-overflow  -Wno-strict-aliasing  -m64  -c src/sudoku_main.cpp -o obj/static/O.darwin.x86_64.gnu.opt/sudoku_main.o
-> linking bin/sudoku.darwin.x86_64.gnu.opt.spx2
g++ obj/static/O.darwin.x86_64.gnu.opt/sudoku_main.o -L/Library/scip7.0.1/scip/lib/static -lscip.darwin.x86_64.gnu.opt -lobjscip.darwin.x86_64.gnu.opt -llpispx2.darwin.x86_64.gnu.opt -lnlpi.cppad.darwin.x86_64.gnu.opt -ltpinone.darwin.x86_64.gnu.opt  -O3 -fomit-frame-pointer -mtune=native    -L/Library/scip7.0.1/scip/lib/static -lsoplex.darwin.x86_64.gnu.opt  -lm  -m64  -lz -lzimpl.darwin.x86_64.gnu.opt  -lgmp -lreadline -lncurses -lm  -m64  -lz -lzimpl.darwin.x86_64.gnu.opt  -lgmp -lreadline -lncurses -o bin/sudoku.darwin.x86_64.gnu.opt.spx2

Sudoku$  bin/sudoku data/puzzle1.txt 
The unsolved Sudoku Puzzle is: 
+----------+-----------+-----------+
|.   2   3 | 4   .   . | 8   .   . | 
|6   .   . | .   .   7 | .   .   . | 
|.   .   . | 5   3   . | 6   2   . | 
+----------+-----------+-----------+
|.   .   5 | .   .   . | .   .   . | 
|8   4   . | .   .   . | .   3   6 | 
|.   .   . | .   .   . | 1   .   . | 
+----------+-----------+-----------+
|.   5   2 | .   9   6 | .   .   . | 
|.   .   . | 1   .   . | .   .   7 | 
|.   .   8 | .   .   5 | 2   1   . | 
+----------+-----------+-----------+
The solved puzzle is: 
+----------+-----------+-----------+
|5   2   3 | 4   6   9 | 8   7   1 | 
|6   1   4 | 2   8   7 | 3   9   5 | 
|9   8   7 | 5   3   1 | 6   2   4 | 
+----------+-----------+-----------+
|2   7   5 | 6   1   3 | 9   4   8 | 
|8   4   1 | 9   5   2 | 7   3   6 | 
|3   9   6 | 8   7   4 | 1   5   2 | 
+----------+-----------+-----------+
|1   5   2 | 7   9   6 | 4   8   3 | 
|4   3   9 | 1   2   8 | 5   6   7 | 
|7   6   8 | 3   4   5 | 2   1   9 | 
+----------+-----------+-----------+

```

Please contact me if you have any issues while running this program.



