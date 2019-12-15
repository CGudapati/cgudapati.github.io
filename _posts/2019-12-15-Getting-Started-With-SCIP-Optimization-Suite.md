---
layout: post
title:  "Getting Started with SCIP optimization in C++: A toy example"
date: 2019-12-15
categories: Integer-Programming
use_math: true
description: In this post, we will discuss how to solve a very simple continuous optimization example in SCIP optimization in c++.
header-includes:
---

I had been working with SCIP for the past week and just wished I had this document when I started. It would have saved me a few hours of needless googling. He, we will see how to solve the following continuous optimization problem:


$$
\begin{alignat*}{2}
&\!\min        &\qquad3& x_{1} + 2x_{2}\\
&\text{subject to} &      & x_{1} + x_{2} &\leq 4,\label{eq:constraint1}\\
& &      2& x_{1} + x_{2} &\leq 5,\label{eq:constraint2}\\
& &      -& x_{1} + 4x_{2} &\geq 2,\label{eq:constraint3}\\
&                  &      & x_1, x_{2} &\geq 0.\label{eq:constraint4}
\end{alignat*}
$$


The above example has been directly taken from [Model Building in Mathematical Programming by H. Paul Williams](https://www.wiley.com/en-us/Model+Building+in+Mathematical+Programming%2C+5th+Edition-p-9781118443330). Before diving into the actual formulation, let us look at a brief installation guide. 

### Installing SCIP 6.0.2 on macOS Catalina with SOPLEX and GuRoBi as LP solvers

We need to compile SCIP from source for this exercise.  Download the latest version of SCIP from [https://scip.zib.de/index.php#download](https://scip.zib.de/index.php#download). Extract the  archived file into a new folder and I renamed the folder to scip6. I then copied the folder to my /Library (notice the leading forward slash). Now my SCIP source code is located in /Library/scip6/ . 

To install SCIP with SOPELX as a solver, let us use the make utility that comes with the source code. In the SCIP directory, 

`\Library\scip6\$ ls`

gives us I highlighted the directories with /

`CMakeLists.txt Makefile    README.md   scip/  ug/ COPYING    Makefile.doit gcg/ soplex/     zimpl/`

To Install SCIP, type the below command and enter

 `\Library\scip6\$ make LPS=spx`

After a few minutes and numerous warnings later, we would have successfully installed SCIP with SOPLEX as a solver. I was also able to install with GuRoBi as a solver on a linux machine. You might face some issues about some libraries READLINE, ZLIB etc missing and you have to compile with those flags set to false. Luckily I did not face those issues, at least on my Mac. On my Debian machine, I had to use apt install to install those and compile it again. 

####  To install SCIP with GuRoBi as a solver:

 `\Library\scip6\$ make LPS=grb`

This will prompt you to enter the softlinks to GuRoBi include and GuRoBi libraries. SCIP tries to help you by

```bash
* SCIP needs some softlinks to external programs, in particular, LP-solvers.
* Please insert the paths to the corresponding directories/libraries below.
* The links will be installed in the 'lib/include' and 'lib/static' directories.
* For more information and if you experience problems see the INSTALL file.

  -> "grbinc" is the path to the Gurobi "include" directory, e.g., "<Gurobi-path>/include".
  -> "libgurobi.*" is the path to the Gurobi library, e.g., "<Gurobi-path>/lib/libgurobi.so" 
  -> "zimplinc" is a directory containing the path to the ZIMPL "src" directory, e.g., "<ZIMPL-path>/src".
  -> "libzimpl.*" is the path to the ZIMPL library, e.g., "<ZIMPL-path>/lib/libzimpl.darwin.x86_64.gnu.opt.a"

> Enter soft-link target file or directory for "lib/include/grbinc" (return if not needed): 

```

But it took me a couple of minutes to try find out where it is. On my machine the include directory is in the following and hit the enter key..

```bash
> /Library/gurobi810/mac64/include
```

```bash
-> creating softlink "lib/include/grbinc" -> "/Library/gurobi810/mac64/include"
```

And the next is for the Gurobi library.

```bash
> Enter soft-link target file or directory for "lib/shared/libgurobi.darwin.x86_64.gnu.so" (return if not needed): 
> /Library/gurobi810/mac64/lib/libgurobi81.dylib
-> creating softlink "lib/shared/libgurobi.darwin.x86_64.gnu.so" -> "/Library/gurobi810/mac64/lib/libgurobi81.dylib"
```

There is no `.so` file for GuRoBi in macOS but it still exists in linux. So the library you are looking for is `libgurobi81.dylyb` and create a link. 

After installing  it, you can check the integrity of the installation by running `$ make test` or `$ make LPS=grb test` depending on the solver installed.

### Building a model in C++  using SCIP library to solve it

Create a directory to keep the project. I call it `SCIP_toyeg\`. In that directory create a `makefile` and  an `src\` directory. In the `SCIP_toyeg/src/` directory create a `main.cpp` file. 

 The makefile looks like this (remove the line numbers as they are only meant for guidance).

```makefile
1	SCIPDIR         =       /Library/scip6/scip
 	
 	
2	#-----------------------------------------------------------------------------
3	# include default project Makefile from SCIP (need to do this twice, once to
4	# find the correct binary, then, after getting the correct flags from the
5	# binary (which is necessary since the ZIMPL flags differ from the default
6	# if compiled with the SCIP Optsuite instead of SCIP), we need to set the
7	# compile flags, e.g., for the ZIMPL library, which is again done in make.project
8	#-----------------------------------------------------------------------------
9	include $(SCIPDIR)/make/make.project
10	SCIPVERSION				:=$(shell $(SCIPDIR)/bin/scip.$(BASE).$(LPS).$(TPI)$(EXEEXTENSION) -v | sed -e 's/$$/@/')
11	override ARCH			:=$(shell echo "$(SCIPVERSION)" | sed -e 's/.* ARCH=\([^@]*\).*/\1/')
12	override EXPRINT		:=$(shell echo "$(SCIPVERSION)" | sed -e 's/.* EXPRINT=\([^@]*\).*/\1/')
13	override GAMS			:=$(shell echo "$(SCIPVERSION)" | sed -e 's/.* GAMS=\([^@]*\).*/\1/')
14	override GMP			:=$(shell echo "$(SCIPVERSION)" | sed -e 's/.* GMP=\([^@]*\).*/\1/')
15	override SYM			:=$(shell echo "$(SCIPVERSION)" | sed -e 's/.* SYM=\([^@]*\).*/\1/')
16	override IPOPT			:=$(shell echo "$(SCIPVERSION)" | sed -e 's/.* IPOPT=\([^@]*\).*/\1/')
17	override IPOPTOPT		:=$(shell echo "$(SCIPVERSION)" | sed -e 's/.* IPOPTOPT=\([^@]*\).*/\1/')
18	override LPSCHECK		:=$(shell echo "$(SCIPVERSION)" | sed -e 's/.* LPSCHECK=\([^@]*\).*/\1/')
19	override LPSOPT 		:=$(shell echo "$(SCIPVERSION)" | sed -e 's/.* LPSOPT=\([^@]*\).*/\1/')
20	override NOBLKBUFMEM	:=$(shell echo "$(SCIPVERSION)" | sed -e 's/.* NOBLKBUFMEM=\([^@]*\).*/\1/')
21	override NOBLKMEM		:=$(shell echo "$(SCIPVERSION)" | sed -e 's/.* NOBLKMEM=\([^@]*\).*/\1/')
22	override NOBUFMEM		:=$(shell echo "$(SCIPVERSION)" | sed -e 's/.* NOBUFMEM=\([^@]*\).*/\1/')
23	override PARASCIP		:=$(shell echo "$(SCIPVERSION)" | sed -e 's/.* PARASCIP=\([^@]*\).*/\1/')
24	override READLINE		:=$(shell echo "$(SCIPVERSION)" | sed -e 's/.* READLINE=\([^@]*\).*/\1/')
25	override SANITIZE		:=$(shell echo "$(SCIPVERSION)" | sed -e 's/.* SANITIZE=\([^@]*\).*/\1/')
26	override ZIMPL			:=$(shell echo "$(SCIPVERSION)" | sed -e 's/.* ZIMPL=\([^@]*\).*/\1/')
27	override ZIMPLOPT		:=$(shell echo "$(SCIPVERSION)" | sed -e 's/.* ZIMPLOPT=\([^@]*\).*/\1/')
28	override ZLIB			:=$(shell echo "$(SCIPVERSION)" | sed -e 's/.* ZLIB=\([^@]*\).*/\1/')
29	include $(SCIPDIR)/make/make.project
  	
30	#-----------------------------------------------------------------------------
31	# Main Program
32	#-----------------------------------------------------------------------------
  	
33	MAINNAME	=	scip_toy
34	MAINOBJ		=	scip_toy.o
35	MAINSRC		=	$(addprefix $(SRCDIR)/,$(MAINOBJ:.o=.cpp))
36	MAIN		=	$(MAINNAME).$(BASE).$(LPS)$(EXEEXTENSION)
37	MAINFILE	=	$(BINDIR)/$(MAIN)
38	MAINSHORTLINK	=	$(BINDIR)/$(MAINNAME)
39	MAINOBJFILES	=	$(addprefix $(OBJDIR)/,$(MAINOBJ))
  	
40	.PHONY: all
41	all:            $(SCIPDIR) $(MAINFILE) $(MAINSHORTLINK)
  	
42	.PHONY: scip
43	scip:
44			@$(MAKE) -C $(SCIPDIR) libs $^
  	
45	$(MAINSHORTLINK):	$(MAINFILE)
46			@rm -f $@
47			cd $(dir $@) && ln -s $(notdir $(MAINFILE)) $(notdir $@)
  	
48	$(OBJDIR):
49			@-mkdir -p $(OBJDIR)
  	
50	$(BINDIR):
51			@-mkdir -p $(BINDIR)
  	
52	.PHONY: clean
53	clean:		$(OBJDIR)
54	ifneq ($(OBJDIR),)
55			-rm -f $(OBJDIR)/*.o
56			-rmdir $(OBJDIR)
57	endif
58			-rm -f $(MAINFILE)
  		
59	$(MAINFILE):	$(BINDIR) $(OBJDIR) $(SCIPLIBFILE) $(LPILIBFILE) $(NLPILIBFILE) $(MAINOBJFILES)
60			@echo "-> linking $@"
61			$(LINKCXX) $(MAINOBJFILES) $(LINKCXXSCIPALL) $(LDFLAGS) $(LINKCXX_o)$@
  	
62	$(OBJDIR)/%.o:	$(SRCDIR)/%.cpp
63			@echo "-> compiling $@"
64			$(CXX) $(FLAGS) $(OFLAGS) $(BINOFLAGS) $(CXXFLAGS) -c $< $(CXX_o)$@
  	
65	#---- EOF --------------------------------------------------------------------
```

This is a slightly complicated makefile but you can use this as a starting point for your other projects. This will compile the SCIP libraries and also the main file. To use GuRoBi as a solver just run `make LPS=grb` or just use `make` if you are using SOPLEX.

Okay. Now for the interesting modeling part. 

```c++
1	//
2	//  scip_toy.cpp
3	//  SCIP_toyeg
4	//
5	//  Created by Naga V Gudapati on 12/13/19.
6	//  Copyright © 2019 Naga V Gudapati. All rights reserved.
7	//
 	
8	 #include <iostream>
9	 #include <scip/scip.h>
10 #include <scip/scipdefplugins.h>
```

We will be including the standard SCIP headers in `scip.h` and `scipdefplugins.h` then we have the main function 

```c++
int main(int argc, const char * argv[]) {
    return execmain(argc, argv) != SCIP_OKAY ? 1 : 0;    
}
```

Now let us see what happens in `execmain()`

```c++
SCIP_RETCODE execmain(int argc, const char** argv)
{
     SCIP* scip = nullptr;
    
    /* initialize SCIP environment */
    SCIP_CALL( SCIPcreate(&scip) );

    
    /* include default plugins */
    SCIP_CALL( SCIPincludeDefaultPlugins(scip) );
    
    SCIP_CALL( SCIPcreateProbBasic(scip, "SCIP_toy_example"));
    
```

We are initializing the SCIP environment in the above snippet. We only create a basic problem by `SCIPcreateProbBasic`and by basic we mean a simple problem. 

To change the objective sense to maximize we can use,

```c++
    SCIP_CALL(SCIPsetObjsense(scip, SCIP_OBJSENSE_MAXIMIZE));
```

##### Adding a continuous variable to SCIP:

We will define a variable  x1 as a simple (basic here) variable and add it to the SCIP environment. As we can see, we can also define the bounds of the variable and its objective coefficient. 

```c++
    SCIP_VAR* x1 = nullptr;
    
    SCIP_CALL( SCIPcreateVarBasic(scip,
    &x1,                       // reference to the variable
    "x1",                      // name
    0.0,                       // lower bound
    SCIPinfinity(scip),        // upper bound
    3.0,                       // objective coefficient.
    SCIP_VARTYPE_CONTINUOUS)); // variable type
                                
    SCIP_CALL( SCIPaddVar(scip, x1) );  //Adding the first variable to scip
    
```

We can also add another simple variable to SCIP similarly. 



##### Adding a constraint to SCIP:

```c++
    1 SCIP_CONS* cons1 = nullptr;;
    
    2 SCIP_CALL(SCIPcreateConsBasicLinear(scip,                 // SCIP pointer
                                        &cons1,               // pointer to SCIP constraint
                                        "cons1",              // name of the constraint
                                        0,                    // How many variables are you adding  now
                                        nullptr,              // an array of pointers to various variables
                                        nullptr,              // an array of values of the coefficients of the corresponding vars
                                        -SCIPinfinity(scip),  // LHS of the constraint
                                        4));                  // RHS of the constraint
    
    3 SCIP_CALL( SCIPaddCoefLinear(scip, cons1, x1, 1.0));      //Adding the variable x1 with A matrix coeffient of 1.0
    
    4 SCIP_CALL( SCIPaddCoefLinear(scip, cons1, x2, 1.0));      //Adding the variable x2 with A matrix coeffient of 1.0
    
    
    5 SCIP_CALL( SCIPaddCons(scip, cons1));
```

Let us add first constraint ($x_{1} + x_{2} \leq 4$). We define a `SCIP_cons` pointer and then create a basic/simple constraint using  `SCIPcreateConsBasicLinear`  in Line 2.  We add it to the SCIP environment and then the more important thing is that we are not adding any variables at this point in the `SCIPcreateConsBasicLinear`. We are just creating a constraint handler.  We are also adding the left hand side as (= -$\infty$) for a <= constraint and the rhs as 4. Then we add the variables and their $A$ matrix coefficient to this constraint in line 3 and 4. In more complex problems, we can define an array of variables and their coefficients and add them.  And finally in line 5) we add the constraint to the SCIP model. Similarly we add another 2 constraints. 

```c++
    SCIP_CONS* cons2 = nullptr;;
    
    SCIP_CALL(SCIPcreateConsBasicLinear(scip,
                                        &cons2,
                                        "cons2",
                                        0,
                                        nullptr,
                                        nullptr,
                                        -SCIPinfinity(scip),
                                        5));
    
    SCIP_CALL( SCIPaddCoefLinear(scip, cons2, x1, 2.0));
    
    SCIP_CALL( SCIPaddCoefLinear(scip, cons2, x2, 1.0));
    
    SCIP_CALL( SCIPaddCons(scip, cons2));
    
    
    
    SCIP_CONS* cons3 = nullptr;;
    
    SCIP_CALL(SCIPcreateConsBasicLinear(scip,
                                        &cons3,
                                        "cons3",
                                        0,
                                        nullptr,
                                        nullptr,
                                        -SCIPinfinity(scip),
                                        -2.0));
    
    SCIP_CALL( SCIPaddCoefLinear(scip, cons3, x1, 1.0));
    
    SCIP_CALL( SCIPaddCoefLinear(scip, cons3, x2, -4.0));
    
    SCIP_CALL( SCIPaddCons(scip, cons3));
```

We now release all the constraints. We can also release them after adding them to the SCIP model. 

```c++
//Scip releasing all the constraints
 SCIP_CALL( SCIPreleaseCons(scip, &cons1) );
 SCIP_CALL( SCIPreleaseCons(scip, &cons2) );
 SCIP_CALL( SCIPreleaseCons(scip, &cons3) );
```

Now we solve the problem. 

```c++
 //Solving the problem
   SCIP_CALL( SCIPsolve(scip) );
```

And now we will create a solution object to take a look at the solution:

```c++
   1 SCIP_SOL* sol;
   2 sol = SCIPgetBestSol(scip);
   3 std::cout << "x1: " << SCIPgetSolVal(scip, sol, x1) << " " << "x2: " << SCIPgetSolVal(scip, sol, x2) << "\n";
    
   4 SCIP_CALL( (SCIPwriteOrigProblem(scip, "scip_toy.lp", nullptr, FALSE)));

    //Freeing the variables
   5 SCIP_CALL( SCIPreleaseVar(scip, &x1));
   6 SCIP_CALL( SCIPreleaseVar(scip, &x2));
    
    //Freeing the SCIP environment
   7 SCIP_CALL( SCIPfree(&scip) );
    
        
    return SCIP_OKAY;
}
```

There is also a `SCIP_CALL` to write the formulated problem into a `.lp` file in line 4 to  verify the contents of the model.  We can also look at the solution from the `SCIPgetBestSol` command. Please email me if you think things can be done better or if they are not very clear.  The full source code is available on my GitHub repository