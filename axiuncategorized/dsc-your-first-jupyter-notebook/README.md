
# Your First Jupyter Notebook!

## Introduction

In this Code Lesson, you'll get your first taste of a real Data Science workflow! You'll use Jupyter Notebook to import pre-written packages, import data into your notebook from a Comma Seperated Values file (a .csv file), and you'll even get to create some plots!

Don't worry if some of these terms are unfamiliar or you get confused. Our goal here is just to give you a taste of what you'll be able to do as a Data Scientist, and afterwards, we'll go back and re-introduce all of the concepts that we skimmed over this time through. By the end of the section, you should have a much stronger understanding of this process - even if you have no previous programming experience.

Just take your time to explore the data and the code!

## Objectives
You will be able to:
* Understand some basic Jupyter Notebooks operations
* Perform cell operations within Jupyter Notebooks
* Use import statements to import and alias Python modules

## Jupyter Notebooks

**Welcome to programming!**
Jupyter Notebooks (historically called IPython Notebooks) will be our primary tool when analyzing data. Jupyter is a web application that allows you to create and work with documents that have live code.  It's a very popular tool among Data Scientists, as it allows for both explanations of thinking behind code as well as the code itself.



## Introduction to cells

The notebook itself consists of cells. Double click on this content in Jupyter notebook to see what we mean. Once you double click on a cell, you are in insert mode. This means that you are able to edit the cells, just as you would if this were a word document. You can tell that you are in insert mode because of the green border around the cell.  

After entering insert mode for this cell, change some content. Don't worry about what you change as you can always undo it. You can revert the changes to a cell by making sure that you are still in insert mode and by pressing `command + z` on a Mac or `control + z` on Windows.

To get out of insert mode and see the effect of your changes, press `shift + enter`.

### Adding cells

If you wish to quickly add a new cell you can do so using the following steps: 

* Make sure you are not in insert mode
    * *Remember you can tell you are in insert mode when you have a green border around the cell*
    * To get out of insert mode, press `shift + enter`.  Another option is to press the escape key 
    * You will no longer see a cell bordered in green 
* Then press the letter `b` to create a new cell 

### Deleting cells

To delete a cell you once again should be in escape mode, and then press the `x` key.

Of course, you'll want a way to undo your deletion.  From escape mode, you can press `z` to undo deletion of a cell.  Note that this is different from `cmd + z`.  Pressing `cmd + z` while in insert mode undoes any changes inside of a cell while, whether these changes be deletions or text insertions.  Pressing `z` from escape mode undoes the deletion of a cell.

Go to escape mode and press `x`. This cell disappears!

Then bring it back with `z`.


### Types of Cells

The current cells and every other cell in this lesson is a markdown cell, meaning that it allows us to write text and stylize that text. For example, if you surround some text with two asterisks (`**`) on both sides, the text **becomes bold**. That's markdown.

Cells can also contain code. If you are writing in a cell that is for Python code, everything in that cell must be valid Python code or you will see an error.



```python
This is a python cell without valid Python code so you will see an error
```


      File "<ipython-input-1-ee7ca47d66cf>", line 1
        This is a python cell without valid Python code so you will see an error
                       ^
    SyntaxError: invalid syntax




Notice that this block cell has a little notes that say **In [ ]:** before you run them, and then are filled with a number after you run them. This is important, as it tells you in what order the cell blocks were run. (Which can affect how a program runs.)

You may also notice that other cell blocks, such as this one, do not have the **In [ ]:** label as with the code blocks. This is because this cell block is formatted as **Markdown** rather than code. The details of Markdown are not important here but just know you can use Markdown cell blocks to display text. They are really useful for embedding notes and explanations in your Jupyter Notebook. You can see (and change) what type of cell is by clicking the dropdown menu at the top:
<img src="images/jupyter_notebook_cell_type_dropdown.png" width="750">

So, a cell must either be of type markdown or of type code, in which case all of the contents must be valid Python.  It cannot be both. We can quickly change a cell from markdown to code with some keyboard shortcuts.

* From escape mode, you can change a cell to type code by pressing the letter `y`
* From escape mode, you can change a cell to type markdown by pressing the letter `m`

Anytime you create a new cell, say with the shortcut key `b`, the new cell will default to code mode.  You can switch to escape mode and press the letter `m` to change the cell from code to markdown.


## Command Versus Edit Mode

You should also start to notice that when you are in a cell writing code (or notes), the cell is highlighted in **green** meaning you are in **edit mode**. 

Alternatively, if you **press esc**, the cursor will be in **blue** inidicating that you are in **command mode**.

### Edit Mode
Edit mode is the standard mode for editing cells, whether it's writing code or notes.
To enter edit mode from command mode simply hit enter, or double click on a cell.

### Command Mode
In command mode, you can delete cells, add cells, copy cells, paste cells, change cell types, and more. You can also do these tasks in a more cumbersome (and time consuming) manner by using the various headers in the menu bar at top.

<img src="images/jupyter_menu.png" width="800">


### View all shortcuts available

You can also see a full list of shortcuts available in command and edit mode under the help menu.

<img src="images/jupyter_help_menu.png" width="800">

There is (of course) also a shortcut to view shortcuts! Press the key `h` while in escape mode to view the menu for all of Jupyter's shortcuts.

## Working with Python in Jupyter

Ok, now that you know a little bit about adding and deleting cells, as well as changing cell types from markdown to code, let's focus on working with Python in Jupyter.  We'll go into a large amount of detail about working with a Jupyter Notebook in Python, but the main takeaway is this: if you see a Python cell, you should press `shift + enter` to execute that cell. 

The major gotcha in working with Python code is that we must execute the cells in order for Python to register the code in them. So for example, just seeing the cell where we define `name` to `'bob'` below does not write that cell to memory.


```python
name = 'bob'
```

If we try to reference that variable later on without having executed the cell, Python will tell us that it is not defined.  


```python
name
```




    'bob'



To execute or run a cell, we must press `shift + enter` on that cell (or when that cell is selected). Upon running a cell, Python will show the the last line of the cell's return value underneath.  Let's run the cell below to see this:


```python
age = 14
age
```




    14



As you can see the variable `age` is set to 14, so when the cell is run `14` is displayed underneath.

One tricky thing to note is that assignment, the action of assigning a variable, **does not** have a return a value.  So, even though the cell is run, if the last line of cell is the assigning of a variable, nothing is displayed underneath. 


```python
hometown = 'NYC'
```

Notice, even after pressing shift + enter on the code above, nothing is displayed below.  But if we reference the variable `hometown`, we see that the cell was run as the variable was defined.


```python
hometown
```




    'NYC'



> Yes, it's pretty confusing, but the important thing to take away is that we need to run our cells with Python code by pressing `shift + enter` if we want Python to read our variables and functions and remember them later on. Remember, in the case of assignment, the return value is `None`, which does not show an output. We can see this more concretely below by running the cell below:


```python
None
```

If you want to print a certain sentence, you need to use the command `print()` and whatever you want to print in quotes, just like this:


```python
print('I want to print this sentence running Python code')
```

    I want to print this sentence running Python code


As you can see, running the code above printed text to the screen. You will learn later that ```print``` is an example of a "function". Functions are blocks of reusable code that execute a particular task. When functions are associated with a specific object they are called "methods." Don't worry about the details right now, you will learn more about functions and methods later.

### Importing Packages
The next thing we're going to do is load in some Python packages that will be part of our toolbox for manipulating and analyzing data. Again, don't worry if you are not sure what a Python package is. For now, just think of packages as collections of specialized tools for performing specific tasks. The standard Python package for working with data tables is called **pandas**. Below, we import this under the **alias** `pd`, which is the industry standard. This will give us a shorthand way to access special functions and methods within the package without having to type the longer name pandas. Similarly, we will also import a specialized package for plotting data called **matplotlib** under the alias `plt`. Pandas and matplotlib will be introduced in detail later. 

* Again, press **shift+enter** to run the code below


```python
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
```

### Loading a DataFrame
Now that you've seen how to navigate Jupyter Notebook cells, you're ready to work with some data. The primary datatype within the pandas package is called a dataframe and is similar to a spreadsheet in excel. Here's a brief example illustrating how to read a csv file from your hard drive and store its contents in a dataframe:


```python
df = pd.read_csv('lego_sets.csv') #Loads the dataframe in
print(len(df)) #Prints the length of the dataframe
df.head() #Uses a built in method common to all Pandas Dataframes
```

    12261





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ages</th>
      <th>list_price</th>
      <th>num_reviews</th>
      <th>piece_count</th>
      <th>play_star_rating</th>
      <th>prod_desc</th>
      <th>prod_id</th>
      <th>prod_long_desc</th>
      <th>review_difficulty</th>
      <th>set_name</th>
      <th>star_rating</th>
      <th>theme_name</th>
      <th>val_star_rating</th>
      <th>country</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6-12</td>
      <td>29.99</td>
      <td>2.0</td>
      <td>277.0</td>
      <td>4.0</td>
      <td>Catapult into action and take back the eggs fr...</td>
      <td>75823.0</td>
      <td>Use the staircase catapult to launch Red into ...</td>
      <td>Average</td>
      <td>Bird Island Egg Heist</td>
      <td>4.5</td>
      <td>Angry Birds™</td>
      <td>4.0</td>
      <td>US</td>
    </tr>
    <tr>
      <th>1</th>
      <td>6-12</td>
      <td>19.99</td>
      <td>2.0</td>
      <td>168.0</td>
      <td>4.0</td>
      <td>Launch a flying attack and rescue the eggs fro...</td>
      <td>75822.0</td>
      <td>Pilot Pig has taken off from Bird Island with ...</td>
      <td>Easy</td>
      <td>Piggy Plane Attack</td>
      <td>5.0</td>
      <td>Angry Birds™</td>
      <td>4.0</td>
      <td>US</td>
    </tr>
    <tr>
      <th>2</th>
      <td>6-12</td>
      <td>12.99</td>
      <td>11.0</td>
      <td>74.0</td>
      <td>4.3</td>
      <td>Chase the piggy with lightning-fast Chuck and ...</td>
      <td>75821.0</td>
      <td>Pitch speedy bird Chuck against the Piggy Car....</td>
      <td>Easy</td>
      <td>Piggy Car Escape</td>
      <td>4.3</td>
      <td>Angry Birds™</td>
      <td>4.1</td>
      <td>US</td>
    </tr>
    <tr>
      <th>3</th>
      <td>12+</td>
      <td>99.99</td>
      <td>23.0</td>
      <td>1032.0</td>
      <td>3.6</td>
      <td>Explore the architecture of the United States ...</td>
      <td>21030.0</td>
      <td>Discover the architectural secrets of the icon...</td>
      <td>Average</td>
      <td>United States Capitol Building</td>
      <td>4.6</td>
      <td>Architecture</td>
      <td>4.3</td>
      <td>US</td>
    </tr>
    <tr>
      <th>4</th>
      <td>12+</td>
      <td>79.99</td>
      <td>14.0</td>
      <td>744.0</td>
      <td>3.2</td>
      <td>Recreate the Solomon R. Guggenheim Museum® wit...</td>
      <td>21035.0</td>
      <td>Discover the architectural secrets of Frank Ll...</td>
      <td>Challenging</td>
      <td>Solomon R. Guggenheim Museum®</td>
      <td>4.6</td>
      <td>Architecture</td>
      <td>4.1</td>
      <td>US</td>
    </tr>
  </tbody>
</table>
</div>



### Python Comments

Note the green text following the pound sign `#`.  

Anything following a `#` in Python is a comment and will *not* get executed. This is a useful feature for annotating your code with notes for yourself and others so that your code is easy to read.

### Accessing Methods from Packages and Objects

A little more theory and context:

When we loaded in packages in our first cell block (using the `import` commands), we loaded code into memory. That code included functions, variables, and other objects. Collectively, all of these items are loaded under the package name or alias.  

We demonstrated this when we used the `pd.read_csv()` method above.  

This also demonstrates the **dot notation** in Python, which is how we access built in methods or attributes of a given object. Similar to what we saw with bash in the command line, we can also use **tab completion** to preview methods available in packages or other objects.  

### Try this out yourself

* In the cell below, navigate your cursor to the right of **pd.**
* Press tab to see a list of available methods/attributes

<img src="images/pd_methods.png" width="600">


```python
pd.
```


      File "<ipython-input-11-bc888235687a>", line 1
        pd.
           ^
    SyntaxError: invalid syntax



You can also start typing to subset the list of available commands:
<img src="images/pd_read_methods.png" width="500">

### Pulling up Docstrings

Even better, you can even see how a method works by pulling up its **docstring**! You can do this by writing `?` after the method and running the cell.

<img src="images/docstring.png" width="800">

Try it out!


```python
pd.read_csv?
```

Alternatively, you can pull up a preview of the docstring by pressing **shift+tab within the parentheses of a method**

<img src="images/docstring_preview.png" width="800">


```python
pd.read_csv() #Move your cursor inside the parentheses and press shift+tab
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-13-59ba97e74ff2> in <module>()
    ----> 1 pd.read_csv() #Move your cursor inside the parentheses and press shift+tab
    

    TypeError: parser_f() missing 1 required positional argument: 'filepath_or_buffer'


### Variables

The other thing that happened in our block of code above was that we defined a **variable**.  

This happened in this line of code:  
> ```df = pd.read_csv('lego_sets.csv')```

As we saw, we used the built in `read_csv` method from the *pandas* package which we imported under the *alias* `pd`. 

The output of this method was then assigned to the variable `df`. This is the standard syntax for declaring any variable. You do not have to specify variable types, as in many other programming languages. Simply:  

> `variable_name = what_to_store_in_the_variable`

### Built-in Python Functions

We also used two built in Python functions:  
- `len() #Returns the length of an object`
- `print() #Prints stuff!`

In general, **Python has reserved keywords** for built in functions like this. **Be sure to not name your variables any of these!**

<img src="images/python_built_in_functions.png" width="700">

You can also check what type of object something is using the built in `type()` method. This can be useful when determining how to work with an object that you are unfamiliar with.


```python
type(df)
```




    pandas.core.frame.DataFrame



### Common DataFrame Methods
As you can see, the variable `df` is a DataFrame object (which is part of the Pandas core package). Here's some other common methods you will want to become familiar with when working with Pandas dataframes:  

- `df.head()` 
    - Preview the first 5 rows of a dataframe. Pass a number for more/less rows
- `df.tail(10)` 
    - Preview last 10 rows (default 5 if no number given)
- `df.info()`  
    - Return column names and details about each column
- `df.columns`  
    - Return column names. Note that there is no parentheses for this. That's because the column names by themselves are an **attribute** of the dataframe, **not a method** called on the dataframe!


```python
#Practice using some of these methods
```


```python
#Use tab completion to investigate at least 3 other methods of the df object. 
#Pull up their docstrings and investigate what they do.
```

### Pandas Series

While the entire spreadsheet is called a **dataframe**, each individual column is known as a **series**. You can access a specific column of a pandas dataframe one of two ways:  

`df['col_name']`

or  

```df.col_name```

First note that in `df['col_name']` we need 'quotations' around the column name. The quotations denote the column name is a **string**, Python's built in variable type for storing text. This can alternatively be replaced with double quotes `df["col_name"]`. In general, anything in quotations is a string in Python code. Occasionally, with very ill formatted column names with quotations in the names themselves, you may even need to wrap a name in triple quotes `df["""col_name"""]` . This will rarely happen in this particular context, but it's the general pattern for dealing with messy strings.   

Note that the second way, `df.col_name`, will only work if there are no spaces within the name of the column. Similar to tab completion with the command line, this is a primary reason why programmers use dashes (-) and underscores (_) in lieu of whitespace in their variable and file names. Also note that no quotations are used when using this format. (The column names have been stored as attributes of the DataFrame object!)


```python
#Previewing Data
```


```python
df.head(2) #preview your spreadsheet again; 2 keeps it a nice small preview
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ages</th>
      <th>list_price</th>
      <th>num_reviews</th>
      <th>piece_count</th>
      <th>play_star_rating</th>
      <th>prod_desc</th>
      <th>prod_id</th>
      <th>prod_long_desc</th>
      <th>review_difficulty</th>
      <th>set_name</th>
      <th>star_rating</th>
      <th>theme_name</th>
      <th>val_star_rating</th>
      <th>country</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6-12</td>
      <td>29.99</td>
      <td>2.0</td>
      <td>277.0</td>
      <td>4.0</td>
      <td>Catapult into action and take back the eggs fr...</td>
      <td>75823.0</td>
      <td>Use the staircase catapult to launch Red into ...</td>
      <td>Average</td>
      <td>Bird Island Egg Heist</td>
      <td>4.5</td>
      <td>Angry Birds™</td>
      <td>4.0</td>
      <td>US</td>
    </tr>
    <tr>
      <th>1</th>
      <td>6-12</td>
      <td>19.99</td>
      <td>2.0</td>
      <td>168.0</td>
      <td>4.0</td>
      <td>Launch a flying attack and rescue the eggs fro...</td>
      <td>75822.0</td>
      <td>Pilot Pig has taken off from Bird Island with ...</td>
      <td>Easy</td>
      <td>Piggy Plane Attack</td>
      <td>5.0</td>
      <td>Angry Birds™</td>
      <td>4.0</td>
      <td>US</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.ages.head()
```




    0    6-12
    1    6-12
    2    6-12
    3     12+
    4     12+
    Name: ages, dtype: object




```python
df.ages[:5] #Here we introduce another new syntax the list slice [:5] this limits us to the first 5 items
```




    0    6-12
    1    6-12
    2    6-12
    3     12+
    4     12+
    Name: ages, dtype: object



### List and Series Slices
Above, we introduced an entirely new programming pattern called a slice which subsets the data into smaller pieces.  
The syntax for a slice is `[start:end]`.   

You can also pass an additional third parameter `[start:end:count_by]` which will allow you to:  
count every other: `[start:end:2]`  
count backwards: `[start:end:-1]`  
or potentially much more cryptic patterns, depending on what you pass.  

While we could have also used `df.State.head()`, slicing works for many more datatypes. This includes the previously mentioned *strings* as well as **lists** and other iterable objects. **Series**, the columns of the pandas DataFrame, are similar to Python's built in *lists*, but also have additional *methods* built in to them that we will continue to investigate.


```python
df.piece_count.
```

### Common Series Methods
Some very useful series methods include those for obtaining basic summary statistics:  
* `series.mean()` 
* `series.median()` 
* `series.min()` 
* `series.max()` 

There are several more but this is a very good starting point.
Start pulling up some docstrings (using shift+tab within parentheses!) and getting acquainted with reading them yourself before we go through a few of these methods together. You can also consult the [documentation.](https://pandas.pydata.org/pandas-docs/stable/reference/series.html)


```python
df.play_star_rating.value_counts() #Read the docstring and tinker! See how it works.
```


```python
#Keep tinkering!
```

### Graphing

Our bread and butter tool for graphing in Python will be **matplotlib**. 
There are a number of other packages you can also check out for visualization later on such as plotly, folium, and bokeh to name a few, but matplotlib is an industry standard used widely. As with the pandas package, we must import  the matplotlib package to use its built in functions and methods. Rather then import the entire package however, we import only the piece we intend to use (hence the dot notation) and again alias this under the shorthand `plt`. Finally, the Jupyter magic command `% matplotlib inline` makes these graphs appear within our jupyter notebook.


```python
#import a subset of the matplotlib package under the alias 'plt'
import matplotlib.pyplot as plt

#ipython magic command for displaying graphs within the notebook
%matplotlib inline
```


```python
to_graph = df.theme_name.value_counts()[:5]
to_graph.plot(kind='barh')
```

### Adding labels
The graph above is a good start, but we should be sure to add some labels! To do this we make successive calls to the `plt` package we imported. Some common methods you should be familiar with include:  
* `plt.title()` 
* `plt.xlabel()` 
* `plt.ylabel()` 

As you might imagine, these methods allow you to add a title and labels to the X and Y axes. 


```python
to_graph = df.theme_name.value_counts()[:5]
to_graph.plot(kind='barh') #lots of other optional parameters can be passed such as color
plt.title('Top 5 Lego Themes', fontsize=16) #add a title and adjust font size (optional)
plt.xlabel('Number of Lego Sets') #you could also pass in fontsize if you wanted here
plt.ylabel('Theme')
```

## Practice and Explore

Take a minute to try chaining some of these methods together to create a few of your own visuals for practice. Some ideas can include making a scatter plot with `plt.scatter(x , y)` where you pass in two series (one as x and the other as y), plotting histograms with the `series.hist()` method, or creating simple bar graphs as shown above.


```python
plt.scatter(df.play_star_rating, df.star_rating)
```


```python
df.play_star_rating.hist()
```


```python
#Your code here
```

## Summary

Congratulations, you got some practice executing cell operations within Jupyter Notebooks and learned how to import Python packages. You also worked through your first real Data Science workflow!
