---
title: Syllabus and Course Policy
layout: syllabus
permalink: /syllabus/
---

(Estimated Reading time: 45 minutes. Please budget an hour so you can also look at the pictures. Though I know this burden is great, I trust University of Chicago graduate students to manage it).

<details class="toc" open markdown="1">
<summary>Contents</summary>

* TOC
{:toc}

</details>



<div class="important" markdown="1">


## \[IMPORTANT\] Course Day and Time: Wednesdays, 2:00-5:00 PM CT

## \[IMPORTANT\] Course Location: Online ([here is the Zoom registration link](https://us02web.zoom.us/meeting/register/bMvS3g7YQaCOVsIxigYIBA))

## \[IMPORTANT\] Course Office Hours:

* Chelsea (Instructor): Fridays, 10 AM CT-12 PM CT  
* Claire (TA): Mondays, 5-7 PM CT except for Week 1, when it will be 5-7 PM on *Tuesday (Oct 6\)*

Office hours entry will be accessible via the **\#intermediate-python-troy** channel in the [MPCS slack](https://cs-uchicago.slack.com/).

## \[IMPORTANT\] Stuff You Need to Get for This Class:

1. Python 3.11 or higher  
2. VSCode and the [Live Share extension](https://marketplace.visualstudio.com/items?itemName=MS-vsliveshare.vsliveshare) for pairing in class  
3. Internet access  
4. A browser (for looking at stuff online—my recommendation is Firefox, but any browser will do)  
   1. I’m not assigning a textbook. Instead, I will refer you to online documentation and resources. They will all be free; if they’re not free, I will make them free for you.  
5. Zoom ([here is the registration link for the class again](https://us02web.zoom.us/meeting/register/bMvS3g7YQaCOVsIxigYIBA))  
6. An account on the [MPCS slack](https://cs-uchicago.slack.com/), to ask questions in the \#intermediate-python-troy channel. If you're enrolled in an MPCS class, you are able to get into this Slack. You can go to [this link](https://cs-uchicago.slack.com/), scroll to the bottom where it says "create an account", and use the email address that's your CNETID@cs.uchicago.edu.  
7. A [Github account](https://github.com/): you’ll be submitting your homework on Github.

## \[IMPORTANT\] Stuff you Should Do for This Class Before it Begins:

* Read this syllabus (congratulations; you’re 2% finished with this one already\!)  
* [Complete the student info survey](https://docs.google.com/forms/d/e/1FAIpQLSd1xZZLkO-FX7PgIunvFduoAaxDEKojEBfE6evpvzMsjlEKVA/viewform?usp=sf_link). I use this to customize the class to each cohort.


</div>

## Course Description 

The MPCS Intermediate Python course provides students with an accelerated introduction to the syntax and features of the Python programming language. 

It *also* endeavors, more importantly in my opinion, to empower students to do what would be asked of them as professional Python programmers: generate Python code to solve *most* of the problems they might face.

The *mechanism* by which junior programmers try to *do* that underwent a sea shift around 2023, after OpenAI launched ChatGPT and as the company’s competitors released a variety of similar products. 

LLM tools do pretty well at generating and analyzing syntactically valid sequences of tokens in programming languages. The smaller the language syntax set and the more examples of the language exist online, the better these tools perform at generating and analyzing code in that language. 

Python usually only has 1-2 right ways to do any given thing (unlike, say, Ruby, JavaScript, or Raku, which offer a lot more…artistic license). Python *also,* by dint of having spent the last 25 years as such a popular language for such a wide variety of end user applications, has the largest publicly available online oeuvre of any programming language besides C and *maybe* JavaScript. This makes it…eminently generatable, really.

**What does it mean to offer an accelerated introduction to Python in a world where programmers can use a SaaS product to generate and analyze Python code for them?** 

I don’t think the answer is to attempt to summarily prevent students from using large language model products, [for reasons I explain here](https://chelseatroy.com/2025/05/14/the-homework-is-the-cheat-code-genai-policy-in-my-computer-science-graduate-classroom/). 

I’ve also been an engineer for…well, a while now, and I’ve been involved with ML for a long time (I’m older than I look—wear sunscreen, folks 😉). I’ve watched a lot of people try to get their jobs done in my field, with and without LLM tools. 

My observation is that students of all stripes—graduate students, YouTube students, bootcamp students—get sold, and prepared for, a *fantasy version* of our job where they *mostly* build stuff by writing code from scratch. 

The problem is, this version of the job hasn’t existed in more than a negligible proportion of the industry since like 1972; before you were born. Before I was born. Before some of your *parents* were born. 

Our job, mostly, involves reading, analyzing, understanding, and debugging *existing* code. We have to be good at *listening* and doing *research* and making *choices.* And generally, computer science education programs do not prepare programmers for that, and interviewers don’t interview for that, so when programmers have to do it (which is most of the time), they’re woefully underprepared, and they feel blocked and unhappy at their jobs. 

The availability of a large language model product to which we can outsource what little “write code from scratch” work we were still doing has not arrested this pattern; it has *exacerbated* it. 

**I think an engineer needs three skill sets to excel at this job.** I think that, if the Intermediate Python class endeavors to empower students to excel as professional Python programmers, it should teach students these three skill sets.

### Three Skill Sets:

1. **Investigation Skills: Asking questions—and then navigating primary sources to find the *correct answers, with high confidence, quickly.*** You do not get this confidence from an LLM product, because answers from those products often depend on a clear and resounding consensus from the internet. ChatGPT will tell you *some reasons why* to use a stack based interpreter. It can even tell you some *likely reasons* that the Python maintainers made Python’s interpreter stack-based. But it notably cannot, with high confidence (I tried it), tell you why, *actually*, the Python maintainers chose to go with a stack-based interpreter. And the Python maintainers’ decisions are relatively publicly and exhaustively well-documented. An LLM product would have even less success at explaining, with high confidence, things like why this function that someone put in your employer’s code base ten years ago includes this ‘if’ clause. You have to be able to investigate that yourself.

2. **Evaluation Skills: Verifying that answers are *correct*, and then understanding the *tradeoffs* associated with those answers.** When we navigate *primary* sources, we can often answer our questions with *high confidence*. When we use *secondary sources* (like biographies, blog posts, or even sometimes real documentation) or *tertiary sources* (sources that amalgamate, parrot, or summarize what was in the secondary sources), we have *lower confidence* in those answers, and we need to verify their correctness before we go off and use them. 

   Sometimes there isn’t even a universal “correct.” Take the idea of *best practices*—an idea that, in programming, usually refers to “the way we all think we are supposed to do it.” The reason something becomes a best practice is that a *lot of people* did it that way and it worked for them. That doesn’t necessarily mean *everyone* should *always* do it that way—it just means that that way works given a set of pretty common circumstances. Whether those are *your* circumstances and whether this solution will work well enough for *you* remains for you to decide. Programmers refer to the details that influence these decisions as ***tradeoffs***. We should know what the tradeoffs are, and we should make the choice that provides us with the benefits we need while exposing us to the fewest drawbacks for our particular situation. 

3. **Innovation Skills: Looking for, and finding, a path forward where there previously was none—*or*, finding a solution with a *better* tradeoff profile for your situation than the solutions tried so far.** This is often the “creative” part of software engineering—how can this piece of software lead to a *better* experience for someone whose life it touches? It is the area in which it is *most rare* for us to be able to simply look up an answer. We often, instead, can only achieve this by *adapting* ideas from other contexts, or *noticing* effects that would be easy to miss, or *synthesizing* multiple ideas from computer science or elsewhere, or even *running our own experiments* on an option that we’ve never—and maybe *no one has ever*—tried before.

How am I supposed to demonstrate these skills to you, or give you opportunities to practice them? For goodness’ sake: I’d basically need some kind of open-source *treasure trove* of well-documented tradeoff decisions, made manifest in code. There aren’t that many code bases like that\! 

This is true. I can count on one hand the number of decades-old open-source code bases I’m aware of whose maintainers have fastidiously documented their dilemmas, their disagreements, and their decisions.

**When I do that one-handed count, though, one of my fingers goes to the CPython interpreter.** 

It’s sometimes called “the standard distribution,” and unless you are downloading a *different* implementation of Python *deliberately*, it’s almost certainly what you use when you write Python code. [Here’s the code of CPython](https://github.com/python/cpython), in full, unencrypted, on Github, in relatively legible C, with pretty good commit messages. 

![Class learning goals: Investigation, Evaluation, and Innovation Skills, each divided into Remember, Understand, Apply, Analyze, Evaluate, and Create](../assets/img/syllabus/image1.png)

## In this class, we will talk about Python syntax, and I will expect you to demonstrate that you can write it.

I *have* to. *You* have to; in order to write this language live, and to analyze a block of Python code with any authority *at all*, you do need to know the syntax. You also cannot pair program or refactor effectively without being able to write and understand Python.

**But we’ll discuss, in addition to the Python syntax, why Python is built the way it is.** You’ll learn about some of the tradeoffs that compiler designers face and why CPython does what it does, as opposed to one of the alternatives. In the process, you’ll get practice traversing a code base and understanding the lines within. You’ll practice thinking about tradeoffs in programming and understanding how to navigate them. An LLM can tell you what the internet *says* to do, but it cannot tell you what *you* should do. This course will prepare you to make those choices. Understanding how something is built, and why it’s built that way, is often a prerequisite to *innovation—*to *improving* upon the status quo. 

It’s also a lot more fun and interesting, in my opinion, than memorizing standard library APIs that you can reference with an internet search. I *need* to prepare you in this class for the professional responsibilities of a Python programmer, but I also *want* to show you what I think makes Python fascinating on its own. 

## Course Calendar:

The schedule for this class is subject to change, but roughly my expectation so far is to do as follows (each number represents a week)

1. Python Syntax Overview, Basic Investigative Steps  
2. Python dictionaries, Tradeoff Analysis, Code Annotation  
3. Exam 1, Python lists, Introduction to parsers and compilers  
4. Functions in Python and Functional Programming  
5. Classes in Python, Functional vs Object-Oriented programming  
6. Exam 2, Overflow day (I would like to look at Pandas on this day if we have time)  
7. Syntactic sugar in Python: where Python uses it and how to implement it ourselves  
8. Parallelism, Concurrency, and Asynchronous Programming  
9. Exam review, Possible special topic  
10. Final Exam  
    

[**You can find the written agenda for each of your sessions on the Schedule page**.](../schedule/) I will try to get each session agenda finalized by about 24 hours before class time; if it is *before that time*, expect the agenda to not be up-to-date yet. **If it is a week or more before the session date, expect the session agenda to *not at all reflect* what we are doing yet.** I use information from your homework performance, exam performance, and session technique survey responses to customize agendas to your cohort.

Here’s the calendar with all your due dates. Transfer it to your favorite calendar app; make it your phone background; print it out and put it in a locket around your neck; tattoo it on your thigh[^1]. Whatever’s gonna work for you.


{% include due-calendar.html %}



<div class="important" markdown="1">

## \[IMPORTANT\] Where to find your homework


![Important](../assets/img/syllabus/image3.png)  
![A Pesquet's parrot](../assets/img/syllabus/image4.png)  
**Your homework assignments will be listed at the *bottom* of the agenda for Session N. This will be the source of truth for the homework due *one minute before Session N+1.***

## \[IMPORTANT\] Grading:

### \[IMPORTANT\] Breakdown (I’ll explain each of these in its own section below):

* Homework: 54 points (6 per class on tasks assigned for completion)  
* Class participation: 54 points (6 per class on discussions and retention questions)  
* Presentation: 30 points   
* Final Project: 42 points  
* Week 3 Exam: 20 points  
* Week 6 Exam: 20 points  
* Final Exam: 40 points

</div>


You’ll be assessed out of 250\. Since the *total points possible is 260,* you will note that you can *lose* some points and still get full marks in this course. In fact, the highest grade I can give you is an A, which I put, you’ll see below, at 235 points. So you can lose 25 points—almost a tenth of the total points possible—in this class before your letter mark is affected. You will also notice that no one *single* graded element of this class can drop you more than 16%, so if you’re just having an awful day on the exact wrong day, your grade isn’t doomed. **You are welcome and encouraged to use all of this leeway wherever and however you need.** You are a graduate student facing a career landscape that will require the wherewithal to make your own decisions; the grading schema reflects that. 

One *example* way to use your leeway would be to get an S—which is a 5 out of 6—on every homework and every class participation (see the subsections of ‘Grading’ about those two components for an explanation of the S-levels), and then lose a *total* of no more than 7 points across the presentation, final project, and final exam. To me, it feels both generous and fair to give that an A effort. 

*Another example way* to use your leeway would be to get an S+ (this is a 6 out of 6\) on all your homeworks, participate fully in class, lose no more than a total of ten points throughout the quarter up until the final exam, and then spend the entire final exam bilious with nerves, turn in absolutely nothing, and walk out a desiccated husk of your former self (-40). Admittedly, this is a somewhat more *aggressive* use of your leeway, but you would still get a B, which is pretty good considering the whole desiccated husk incident.

A C+ or higher is a passing grade in this course, so my points taxonomy chart does not descend beyond that. Points totals below are already rounded and are firm (as in, accumulating 224.5 points does not bump you from a B+ to an A-). 

| Letter grade | Percentage |
| ----- | ----- |
| A | 94–100% (minimum 235 points) |
| A− | 90–93% (minimum 225 points) |
| B+ | 87–89% (minimum 218 points) |
| B | 83–86% (minimum 208 points) |
| B− | 80–82% (minimum 200 points) |
| C+ | 77–79% (minimum 193 points) |
| C | 73-76% (minimum 183 points) |
| C- | 70-72% (minimum 175 points) |

You will notice that the minimum passing mark in this class requires you to earn 175 points, which means you could lose 85 points out of 260 and still pass.


<div class="important" markdown="1">

#### \[IMPORTANT\] Requests for leniency


![Important](../assets/img/syllabus/image5.png)  
![A secretary bird](../assets/img/syllabus/image6.png)

**I will not be receptive to requests for additional leniency beyond the leniency policy explicitly stated in this syllabus and course policy document. I will not even guarantee you a response, from me or my course staff, to messages requesting additional leniency beyond what is stated in the syllabus.**

</div>


 **“Why?” Great question. Reasons, in descending order of importance:** 

1. **I want to protect your capacity to pay attention to the things that matter in this class.** This is the essential function of a course policy. I take “asking for leniency” *off* the table so that there is more room *on the table* for you to focus on the course material.   
2. **Random, one-off leniencies afforded upon request are not fair to students.** I’d like you to know all your options up-front, so there are no “hidden doors” accessible only by asking the instructor.  
3. **My existing leniency policy is already quite generous.** I’ll treat your continuing in this course beyond week 1 as your agreement with this statement, just as if you had put a signature of acknowledgment on the bottom of this syllabus.  
4. **The prospect of leniency upon request incentivizes students to tax instructors’ emotional capacity.** If students think they’re more likely to get leniency by convincing an instructor that something horrific happened to them, that incentive encourages a broad *pattern* of instructors hearing a larger number of more horrific stories. 

   We care deeply about each of you. But we, like you, are *people,* and that’s a lot to put on us. I am also, in addition to being a person, a Mozillian. Personal privacy is my profession. I am [not interested in accepting compromises to your personal privacy](https://tenureshewrote.wordpress.com/2017/06/19/to-my-colleagues-on-the-death-of-their-students-grandmothers/comment-page-1/) in exchange for points in my class. 

   If you come to me with a traumatic event during the quarter, I am happy to help you access University resources including a social worker, guidance counselor, medical professional, or therapist. But there cannot be a *grade incentive* for starting that conversation.

Sometimes, things happen in our lives, and we need to take a step back and focus on our health, our family, or another core value ahead of, or instead of, achievement in graduate school. One of my goals as a mentor and instructor to graduate students such as yourselves is to help you clarify your values and act in accordance with them, so it should come as no surprise that I fully support you in making these decisions. **However, these situations do not entitle students to changes in the course policy to allow their grade to reflect their original intentions for achievement in graduate school.** 

**Before we continue, a piece of advice from me to you: I would highly recommend *against* attempting to launch a startup or begin your first ever full-time job during a quarter in which you are taking a class that you *must pass* in order to remain enrolled in the program next quarter.** For a lot of people, your core programming requirement is one such class. I have watched *both* of these things, undertaken in a quarter where students cannot *afford* to fail a particular class, affect students’ chances of proceeding through the program at the rate they would like.

#### Homework: {#homework}

The homework grade is designed to support your development of **Investigation Skills and Evaluation Skills.**  
![Pie chart of the three skill sets with Investigation and Evaluation Skills highlighted](../assets/img/syllabus/image7.png)

I’m aiming for the homework in this class to take 7-9 hours per week completed at the S level (S-levels come from [specifications grading](https://teaching.unl.edu/resources/alternative-grading/specification-grading/), and I’ll explain how it applies to your homework below). Most assignments will include: 

* **Reading:** Some of the *best resources* for learning Python are the Python documentation, the online writings of the maintainers, and *select* books about general programming language design and Python’s specific design (not *all* books on this are good; I picked you good sections of excellent books ;).  
* **Writing Code:** Most of the code you write for homework is going to focus on the **what**—what to write in Python in order to get your thing working. In class, we will talk about **why** we do it that way as well as how it works.   
* **Reflecting:** It’s valuable for you, as a self-directed learner, to reflect on how you reached the conclusions you did. It’s also valuable for me, as your skills coach in Python, to understand this information, so I can help you build your skills and improve my assignments to better exercise your skills.


<div class="important" markdown="1">

##### \[IMPORTANT\] Homework deadlines and late work

 ![Important](../assets/img/syllabus/image8.png)  
![A king vulture](../assets/img/syllabus/image9.png)

**Homework is due one minute before the session *following* the one in which it was assigned (so, for our regular sessions, homework assigned in Session N is due at 1:59 PM on the following Wednesday, one minute before Session N+1 begins). See the “Schedule” figure for a visual representation of this**

**In this class, “turned in” \= “there is a PR on Github in your homework repo with the homework in it, completed in full.” Homework done on your local machine and not in the repo? It’s not turned in, and turning it in after the due date does not constitute an on time submission.** 

If something comes up and you need some breathing room in a given week, you may turn in your homework up to 48 hours *after* it’s due for one point off (for example if your work is turned in this late and your effort merits an S, you’d get an S-). The late deduction is not prorated: it’s a whole point off whether the thing is 8, 18, or 39 hours late. **Beyond 48 hours late, it’s a zero.** 


</div>

**Each homework is worth six points. The six points correspond with the following marks:**

* 6: S+, which stands for “more than satisfactory.” Excellent work, exceeds my general expectations for this assignment, demonstrates particularly thorough insight or reflection. I’m impressed.   
* 5: S, which stands for “satisfactory.”Meets my general expectations for this assignment.” You did it all, you did it well, and we’re happy with this effort.   
* 4: S-, which stands for “somewhat less than satisfactory.” Might meet the *letter of the instructions* for the assignment, but fails to meet my expectations for quality of reflection.  
* 3: U, which stands for “unsatisfactory.” Does not meet the *instructions* for the assignment.   
* 0-2: [Specifications grading](https://teaching.unl.edu/resources/alternative-grading/specification-grading/) doesn’t quite have an equivalent for this, but consider this the range you can expect for incomplete or missing assignments.[^2]

A note on S levels: An S+, or full marks, is “impress me”—it’s not just “did the letter of the assignment,” which you’ll see in the above taxonomy is a mere S-. This means that to get full marks on every homework, you’d have to impress on every single one, and since “impress” is definitionally up to course staff discretion, *I do not expect this of you*. I do not think you should try to do this, or feel like a failure if you do not achieve it. The leeway to get just an S on things a lot of the time without suffering a final grade consequence is the reason this class has 260 points possible but is graded out of just 250\. You could get an S on every participation and homework grade and still have 7 points left over before falling out of the ‘A’ category. 

**Regrade Opportunities:** 

[Specifications grading](https://teaching.unl.edu/resources/alternative-grading/specification-grading/) includes *regrade* opportunities so that you receive points for incorporating feedback. Our regrades in this class work like so: 

* Your original submission has to have been an on time, good faith effort to complete the homework. “Good faith” is subject to course staff discretion and not arguable. Late submissions are not eligible for regrades.  
* Your homework must have received a grade of a 3 (U) or a 4 (S-) to be eligible for a regrade. An S (5) is plenty good enough: you do not need a regrade for this. Below a 3 does not constitute a good faith effort.  
* The highest mark a regraded assignment can receive is a 5 (S).  
* To request a regrade, contact the member of the course staff who graded the assignment for you originally and request one. Please give us a week from your request to get it regraded for you.  
* Each homework can receive a maximum of one (1) regrade. 

#### Class participation: {#class-participation}

The class participation grade is designed to support our development of **Investigation Skills, Evaluation Skills, and Innovation Skills.**

![Pie chart of the three skill sets with all three highlighted](../assets/img/syllabus/image10.png)

Class sessions are mandatory in this course. If you are unable to attend an individual class session, please notify the instructors and course staff ahead of class, and we’ll give you 2 of the 6 participation points for that day (otherwise it’s a zero). There is no way to get full participation credit for a class that you cannot attend, even if you have a legitimate reason.[^3] To get full credit for class participation you are expected to come to each class and complete the activities therein. These can include:

* Individual activities: writing reflection exercises, asking questions   
* Paired activities: close reading and reflection exercises  
* Small group activities: programming challenges, diagram creation and annotation, discussion  
* Large group activities: group discussions and annotation exercises.   
* Pairing with the instructor in front of the class: in this class I may ask you, specifically, to step up and pair with me on a programming problem. Please be prepared to be on camera, with your audio working, for any class, such that this is possible. 

Each class participation grade will *also* do the six-point specification grade thing described in the homework section above. 

“But I am a quiet person and have trouble breaking into conversations”—heard. *I will be careful in this class to provide written, simultaneous contribution mechanisms* (like diagrams where you will have a color that represents your work) and *individually assessable contributions* (like when I call on you to ask questions of your peers and when I give you retention questions to complete individually). I also usually have group work happen in groups of no more than one person per two minutes of exercise time, with a max of four people, matched by caucus score. These mitigations protect your opportunities to contribute and demonstrate participation in the class.

#### Presentation: {#presentation}

The presentation grade is designed to support your development of **Investigation Skills** and **Evaluation Skills.** 

![Pie chart of the three skill sets with Investigation and Evaluation Skills highlighted](../assets/img/syllabus/image11.png)

You will be responsible for presenting to the class (*maximum time* ten minutes) about a library or set of libraries. Part of your homework for Session 1 will be to submit 4-6 candidates that you would be interested in presenting on. You’ll select 2-3 options from each of these two groups:

**Group 1 (Modules from the Python Standard Library):**

* os and sys (together)  
* Math and random (together)  
* time and datetime (together)  
* re and string (together)  
* Json and csv (together)  
* Collections and itertools (together)  
* pathlib and logging (together; this is the only pair of libraries where the purposes of the two are kind of unrelated, but I needed to put these last two together to form a pair of appropriate scope)  
* Unittest and Pytest (both; this is a weird one because Pytest actually isn’t in the Python standard library, but these two libraries deserve a direct comparison)


**Group 2 (Extremely popular third party libraries):**

* Numpy and Pandas (together)  
* Scipy  
* Matplotlib or Plotly (I want a presentation on one or the other of these but probably not both)  
* Scikit-learn   
* Keras and Tensorflow (together)  
* Pytorch  
* Flask and FastAPI (the presenter would compare the two of these because the fundamental API for each is quite small)  
* Django 


As long as there are a minimum of 2 options from each of the above groups, I am okay with you also *proposing* a library *not* in this list that you’d like to cover. Know that you’ll have to convince me of a good reason why the class should be asked to watch a presentation about your favorite library, and that I am pretty likely to *not* assign you the one you proposed because *I* think the ones on my list are pretty important already. If multiple of you are hell-bent on doing the same specific library and that library is large enough to split up, I’ll consider assigning you each a specific aspect of that library. The idea here is to give you *some amount of input* into *my decisions* about who gets to present about what, and when.

I’ll take those suggestions and preferences into account as I assign you a library option for the presentation. I will endeavor to ensure that each library option is only getting one (1) presentation per quarter. 

You will construct a presentation that fulfills the following requirements, presented via video, uploaded to either Panopto or YouTube: 

* Content: 27 points (examples cover [git-revise](https://git-revise.readthedocs.io/en/latest/), a dependency written in Python that is *not* one of your options to do a presentation on):  
  * Introduces a *case* that helps *motivate* what you might use this library or each of these libraries for (example: “suppose we made some commits, but right now they’re out of order. We want to change the order in which these changes are committed, and also move some of them to other commits” (3)  
  * Introduces your library or libraries and how it/they can help with the example problem (“Here is a list of commits I have made. When I do this command in git-revise, it allows me to reorder these commits. The same operation done with regular old rebase takes these *additional* commands and runs a lot slower, especially when the commits are kind of far back.”) (3)  
  * Demonstrates several of the main functions of your library/ies (9)  
  * If the library/ies has *more* functions than the ones you demonstrate, your presentation gives an idea of what those are and what else the library/ies might be used for (3)  
  * After your presentation the class asks you questions about your library/ies, and you answer them. Answers to class questions should demonstrate a firm grasp on the material (9 points).  
* Format: 3 points:  
  * 10 minutes in length, hard maximum, recorded and uploaded at 1x speed. If it’s *less* than this, that is okay, as long as it fulfills the content requirements (3).   
    * I know I’ve labeled this 3 points, but if the content is *wildly* over 10 minutes (like it’s 25 minutes or something) I reserve the right to take more points than this. The objective of this assignment is “provide a terse, well-organized introduction to this library/ies that would help a Python programmer recognize situations when they should reach for this.”    
* Delivery:  
  * We need to be able to hear you and understand what you’re saying. I’m not grading you on charisma, performance, or bombasm—this is not a public speaking class—but you need to be able to get your message across. I trust the ‘Content’ portion of the rubric to cover this requirement.   
  * Chelsea will assign you a library option and week in which your presentation is due. Because this class is designed to have students submit and then *review* these videos in small batches, some students will have more time to prepare than others. To mitigate this, students who present in week 4 will get a \+4 to their presentation score, week 5 \+3, week 6 \+2, week 7 \+0. These points are not capped at full credit for the presentation: if you get a 30/30 on your presentation and also get 3 bonus points, you still keep all 3 bonus points toward your final grade.


<div class="important" id="presentation-questions" markdown="1">

##### \[IMPORTANT\] Questions about classmates' presentations


![Important](../assets/img/syllabus/image5.png)  
![A shoebill](../assets/img/syllabus/image12.png)

**During several of the weeks of this class, your homework will include watching several of your classmates’ presentations and then submitting a question about each presentation**. 


</div>

**What makes a creditworthy question? It needs to:**

* **Be novel to the discussion:** I will not award credit for asking a question that has already been asked. I also will not award credit for asking a question that the presenter answered verbatim in the presentation itself.  
* **Be specific to the topic:** I will not award credit for asking a question that could plausibly be asked about *a wide selection of* topics, because those don’t prove to me that you paid attention to the presentation. Examples would be:  
  * “How has AI impacted problems solved by \[library\]?”  
  * “Do you think \[library\] is still relevant to understand today?”   
* **Introduce new factual information:** The question should leverage the presenter’s knowledge and research to teach you more information about the topic. Questions that ask the presenter to *speculate* or share their *opinion* don’t reliably do this, so I recommend avoiding them. Examples might be:  
  * “How do you anticipate that AI will impact \[library\]?”  
  * “What is your opinion of \[library\]?”  
  * “What is your personal prediction about \[library\]?”

I will ask you to submit it via a form where you can see what has been submitted so far. The presenter will then be responsible for answering these questions as part of their homework, due before the beginning of the following session just like all the other homework. The presentation grade includes 9 points devoted to answering these questions thoughtfully, completely, and (to the extent reasonable for the level of research I’ve asked of you) correctly. 

From these question-answer pairs, I will select candidates to be tested in exams based on whether I think the question covers useful material and whether the presenter’s answer was correct enough to count for the points on the final exam. 

#### Final Project: {#final-project}

The final project grade is designed to support your development of **Investigation Skills, Evaluation Skills, and Innovation skills.**  
![Pie chart of the three skill sets with all three highlighted](../assets/img/syllabus/image13.png)  
You will be responsible for engaging with the skills and topics we discuss in this class by *creating* something using Python. 

**COMMON QUESTION:** “Can I do a project that includes languages other than Python? For example, can I make a website that uses HTML and CSS?”

**ANSWER:** Yes, but the non-Python portions do not contribute to your grade; only the part that specifically concerns Python does. For this reason, I don’t really *recommend* doing a website: you’ll find a much higher ROI from a grade perspective if you make a [Python package](https://py-pkgs.org/04-package-structure), for example, or do a data analysis project that purely uses Python. 

Part of your homework for Session 2 will be to submit 2-3 candidate project prompts and execution plans, on which the course staff will provide you feedback.

Examples of interesting final project prompts:

* I would like to try to make an “LLM” with just a Random Forest Classifier and see how big/complicated it has to be before I get something halfway plausible out of it (I’ll come up with performance metrics that I’ll use to define “halfway plausible.”). I’d like to track the compute required to train it and the size of the final model and compare that to RNN and transformer language models.  
* I would like to make an illustrated guide, in the style of a children’s book, that introduces several of the different algorithms at work in the CPython implementation and why they were chosen. I will use carefully designed diagrams and explanations to inspire Python programmers to understand how algorithms affect the programming language they use every day.

If you want to work in *pairs* or *groups* on a final project, you may do that, but each of you must have a *distinct portion* that *you are managing.*

Example of a joint project prompt:

* ![Bert](../assets/img/syllabus/image14.png)**Bert will:** find out what data Chicago has about bike lanes and bike usage, compare that to information from other cities, perform documented data analysis to guide decisions about a bike lane placement algorithm, and identify the tradeoffs of those decisions as well as how to mitigate them. Bert will also write the unit tests for the bike lane placement algorithm and investigate automatically updating documentation solutions for the code.  
* ![Ernie](../assets/img/syllabus/image15.png)**Ernie will:** Find, and integrate with, a map API to illustrate suggested bike lane routes through Chicago. Ernie will embed this map into a UI that permits consumers to change the input parameters to the algorithm and see the map change in real time. Ernie will also select a web framework and deploy the visualization and interactive elements on a webpage with it. Finally, Ernie will implement a feedback mechanism by which consumers can submit comments on the bike route suggestions and the maintainers can see those comments in an aggregated location.   
* ![Bert](../assets/img/syllabus/image14.png)![Ernie](../assets/img/syllabus/image15.png)**Bert and Ernie together will:** Prepare the README and contribution guidelines for their algorithm, write the copy explaining the project for the webpage, and script and present the work in the final video. 

With each prompt proposal you will submit an execution plan that describes what you intend to do on your project at each of *three* checkpoints (the week of session 5, the week of session 7, and the final project due date). It should cover, at a high level, all of the tasks you’ll need to complete. 

Imagine that your *boss* assigned you an ambitious project, or your PhD advisor came to you about a project, and this authority figure wanted to know how you would approach completing the project, and you anticipate a professional reward for convincing this authority figure that you know what you need to do. *A*n S+ level execution plan will tell us what you plan to do, how you plan to do it, and why that step is important. If your execution plan is, say, one line for each week, expect an S- at best.  

**This is an example of an S level Final Project Prompt and Execution Plan:**

* **Prompt:** I like to shop secondhand for things, because I know that buying already-made things is easier on the planet than producing new things. The problem is, it’s hard to find the exact things I need this way. So I would like to make an app that aggregates listings from online secondhand portals and also allows me to place watches on particular keywords, so I will be notified if such an item comes up.

* **Execution Plan:**  
  * **By Week 5 Homework Deadline:**   
    * **Find at least five different sources for secondhand stuff that I can query via some kind of API or scrape.** I will check Poshmark, ThredUp, thriftbooks, Ebay, Salvation Army Online, Goodwill Online and the Buy Nothing Project for APIs or scraping options. I will also see if I can integrate with Facebook marketplace.  
    * **set up an endpoint that accepts a search query, and have that endpoint fetch using the search query from one of my APIs I picked out.** I plan to use FastAPI or Flask to set this up. I am not bothering with account creation for this project: the data stays local on the consumer’s phone, and if they get a new phone they start over. I can worry about accounts later.   
  * **By Week 7 Homework Deadline:**   
    * **Make the list view of my search results prettier in the app, and also add in my other four APIs that I picked out before the last checkpoint.** So now I should have an API where I can search for a keyword in my search bar, and it gives me results from all of my thrifting sources. For now I’m not going to do *purchase* integration: I’ll just give the user a link that opens in a browser to the page online for the product.  
    * **Let consumers of my API place “watches” on specific keywords.** For example they could place a watch on “dutch oven,” and each day my app will run that search and notify the consumer about new “dutch oven” listings since the last time they opened the app, maybe via an email integration.  
    * **Add running totals for money and environmental impact metrics:** I’d like my consumers to be able to mark their “to buy” list items as “thrifted” via an endpoint, and I’d like to make an endpoint that shows a total of some environmental and cost metrics they’ve *saved* (like maybe “Saved \$X and Y kg CO2 by thrifting\!”) To do this step I have to 1\. Figure out some good metrics to use, 2\. Figure out how I would measure this, and then 3\. Make an endpoint that accepts their email address for calculating their specific metrics.  
  * **By the final due date:**   
    * **Make things prettier, or fast if they’re too slow, or fix issues.**   
    * Once I’m generally satisfied with the API design and behavior of my app, if I have time left over, I’ll go ahead and implement login and accounts. This would require me to set up an API server and then figure out authentication, which is not the point of this project, and is why I am leaving this for *last*.   
    * On Thursday morning in week 9, I plan to film my video.

<a id="final-project-grading"></a>

Now, those are some pretty specific checkpoints. The *reason* for those checkpoints is that t**he total points for your final project are assessed *throughout the quarter as follows*:**

* 14 points of your final project will be assessed in week 5, based on your progress relative to your execution plan (10 content, 4 delivery)  
* 14 points of your final project will be assessed in week 7, based on your progress relative to your execution plan (10 content, 4 delivery)  
* 14 points of your final project will be assessed after the final deadline, based on your project’s fulfillment of the following requirements: (8 content, 6 delivery—the breakdown changes because the final deadline is where we see your video)

**What’s creditworthy for the final project? In each of the content and delivery categories respectively, it’s this:**

* Content:   
  * Project includes sufficient scope of work (28 points total over the quarter)  
    * You’ll get feedback on scope when you submit your proposal.  
    * It’s a 7-week project, not a doctoral thesis. We will keep this in mind. Any type of “study” you undertake, we’ll expect a “preliminary exploratory study” level of complexity and rigor, not an “academic paper” level of rigor.  
* Delivery:   
  * Artifacts that convince me you’ve done the portion of the project you told me you’d do (14 points total over the quarter).  
    * Clearly, with projects being so open-ended, these could be a lot of things. For the project proposals above in order this might be:   
      * A code base including your different tries at the “Random Forest LLM,” with a summary of the results you got upon running each one  
      * The illustrated board book—ideally what it looked like at a few different stages of completion.  
      * Bert’s bike lane data, the analysis code, the visualizations, and the conclusions.


<div class="important" markdown="1">

##### \[IMPORTANT\] Final project video


![Important](../assets/img/syllabus/image16.png)  
![An African jacana](../assets/img/syllabus/image17.png)

* **At the very end (after the final deadline), your final project artifacts must include a video that demonstrates your code working and also explains your code to us. We have this step in here for three reasons:**  
  * Gives you practice presenting your technical work  
    * You have an artifact a potential employer would look at (they are not going to bother to browse your repo, sorry)  
      * Prevents us from having to dock points if your project does not run on our machines; we can look at the video and tell if it ran fine on *your* machine. Successful replicability of a software project is, while an important engineering skill, not one of this course’s pedagogical goals.

        
        This step is only for the *final* delivery; you don’t need to film a video at the checkpoints.

      


</div>

**For this final project video, the delivery requirements are as follows:**

      

* The video should include three things:  
  * Show us how the final project works (demo)  
  * Show us the code: what it does and how it’s organized  
  * Talk to us about tradeoffs you faced during implementation. How did you decide to execute, and why?  
* The video should be *no longer than twenty minutes,* played at 1x speed, without being artificially sped up ahead of submission.


Final project checkpoint grades are not eligible for regrade.

#### Exams: {#exams}

Exams will be administered three times.

Exam portions that test recall with facts and information will have a *deliberately* tight time window. Exam portions that test critical thinking will have a time window as well, although less tight. 

##### How will the questions be selected? 

The first exam is pretty simple—I’ll select the questions. Those questions will cover Python syntax concepts you’ve read about in your homework, plus implementation details we’ve discussed in class. If you’ve done the homework and attended class, you should find this first exam to be *pretty easy*. If you do *not find this exam to be pretty easy*, we need to have a conversation about your readiness for an accelerated programming class.

For the second exam and the final exam, you will have an opportunity to contribute to the questions.

The science of learning tells us that **three steps** must happen for learning to occur: 

1. **Attention:** Students need to focus on the information, while filtering out other stimuli  
2. **Processing (sometimes called Encoding):** Students need to interpret and make sense of the information  
3. **Retrieval:** Learners need to access the information after they have processed it.

Exams, chiefly, support retrieval. 

Different retrieval tasks demonstrate different levels of mastery of the material, first classified (that we know of) in the 1950s by B. Bloom and his team into six different levels. These levels have been [recontextualized](https://www.jstor.org/stable/20385884), [revised](https://www.jstor.org/stable/42926529), and [critiqued](https://www.lifescied.org/doi/10.1187/cbe.20-08-0170) extensively in education research since then; here is a helpful diagram from Oregon State University that defines and exemplifies the taxonomy’s 2001 revision:  
![Chart of Bloom's taxonomy levels from Remember to Create, comparing AI capabilities with distinctively human skills at each level](../assets/img/syllabus/image18.png)

“Remember” is the *least advanced* level of retrieval. “Create” is the *most advanced* level of retrieval. The effectiveness of an exam at supporting a learner’s retrieval depends on asking questions that give learners the chance to exercise each of these retrieval levels. 

Conveniently for *your* learning, one effective tactic for empowering you to execute at the *most* advanced retrieval level (“Create”) is to empower you to *create* questions for a theoretical final exam. 

Throughout the class you will have opportunities to propose questions together for the final. To the extent that I like your questions and the way that they exercise the different retrieval levels, I will select questions for the final exam out of the questions you generate, which you will be able to see in full in a spreadsheet.

I will also generate some portion of the questions on the exams myself from the material in this class.

I am not yet certain what the proportion of each will be: this will depend on my evaluation of your questions. The more I like your questions, the more of the exams comes out of your questions. The less I like them, the more of the final comes out of *my* questions. You will notice three rounds of “Synthesis Questions” activity in the course map. I will give feedback on your questions after each round, so that in the next round, you can use that feedback to write a higher proportion of questions that I am likely to use for your final exam.

#### Chelsea, why do participation and homework use [specifications grading](https://teaching.unl.edu/resources/alternative-grading/specification-grading/), and the rest of the class does not?

**Great question. Here’s why:** class and homework are chiefly about *practice*. The granularity (and in the case of homework, mutability) of their grades should affect that. I want effort and engagement from you as *learners*, and I think [specifications grading](https://teaching.unl.edu/resources/alternative-grading/specification-grading/) suits that pedagogical goal.

Your presentations and projects are instead about *performance:* you’re not just a learner in your execution of these tasks. You are also a teacher and a contributor. The quality of your output on this work affects not just you, but also your classmates and anyone whose life your project affects, or could affect. For that, I want more granularity.

Exams are a weird case. In the exam you’re in the role of *learner*, but I’m evaluating your performance rather than your effort, so its pedagogical orientation falls right down the line. I go with traditional grading because it’s easy to execute for course staff and well understood by students, who regularly take traditionally graded exams.

Exams are not eligible for regrade.

## How to succeed in this class:

This graduate program contains some of the brightest minds in the world, all training to work in a pretty self-directed field. So I am 100% comfortable expecting and evaluating for a *convincing* good faith effort on each assignment in this class, and exercising my judgment on the level of quality and reflection I see. **I want to emphasize for you that this is exactly how you are going to be evaluated in a professional career.** No one is going to hand you a rubric that offers you an exact number of points for each specific thing you do. You’ll be given job offers, raises, promotions, and even your PhD (should you choose to go for one) based on your ability to convince people to advocate for you in circumstances of *murky rubric criteria*, if the rubric exists *at all*. You can probably pass this class doing the minimum possible interpretation of each task I assign. You won’t get 100s, which is fine if that’s not what you’re going for, but I’ll refer you to this paragraph of the syllabus if you come to office hours and ask why that’s the case.

​​As far as “how I recommend you approach the material in this course,” my thoughts on this, broadly-speaking, remain well-circumscribed by [a deliberately short guide I put together as an undergraduate to help my teammates on a Division 1 athletic team to succeed at their coursework](https://docs.google.com/document/d/1y4KnksE5xJI7W3UUME7NZ-wCOPXQH-HwKn4ckl3lt6g/edit?usp=sharing). I wrote this guide 15 years ago and my writing style at the time was a bit cringey. In a few places the advice selects for an undergraduate audience (I have added comments to point out where this is the case). Nevertheless, I stand by the recommendations I made in this document, or the graduate school adaptation thereof. 

## Technology Policy:

This class takes place online\! 

### Why do we have a remote class in an in-person program? 

**Great question. Reasons, in descending order of importance:** 

1. **Remote collaboration is a skill set that you require to thrive and succeed in tech.**   
   1. Even in this Mad Men cultural era where Amazon and the Fed and whoever are calling people back into the office, about 40% of tech jobs are still remote.  
   2. Even the people who go into the office need remote skills to succeed. Full-time-in-person tech workers still regularly pile into meeting rooms where the other half of the meeting is some faraway client on Zoom. They still have Slack, take meeting minutes in Google Docs, and organize projects by all updating a shared spreadsheet. They still do pull request review on Github, and even the pejorative term “email job” exists *because* most of the *real work* in those jobs is accomplished by communicating with people *via email*. These are all remote skills, and their relevance is not going away no matter how many managers kick and scream and issue RTO orders.  
2. **Remote collaboration and execution is even *more* likely to be necessary in your pursuit of positive impact in your career.**  
   1. You want global impact? In this age, that probably means effective, continuous, deep collaboration with people around the globe. (We’re not even practicing *time zone differences* in this course, which is a higher-level skill in the remote collaboration skill set).  
   2. For this class to adequately prepare you for a high-impact career, it’s beneficial to help you strengthen this skill set.  
3. **The remote format makes a lot of our collaborative exercises easier.**  
   1. I believe that pair programming is an excellent way to practice your problem-solving skills, both in class and in the workplace. However, comfortable in-person pairing requires a specialized setup with two mirrored monitors. We don’t have classrooms like this at UChicago. I do not want four people hunched over one person’s laptop. This is not a LAN party circa 1998\. On Zoom, one person can share screen and the other groupmates can navigate, no crowding required.  
   2. I use small group work to allow you to tackle challenges together. Several small groups working on the same problem in a room together *can* (don’t always, but often can) distract one another and interrupt one anothers’ trains of thought. Furthermore, the stadium seating classroom where everyone faces one direction, each row is one long desk, and all the chairs are nailed to the floor does not support this kind of work. Breakout rooms facilitate it, though.

**Remote work also makes a number of things harder:**

1. It forces you to look at the screen of your computer, a device on which you probably have several apps whose user experience designers have deliberately built in notifications to pull your attention to the app (and away from the classroom).  
2. Looking at a screen for an extended period, and managing your appearance at close range on a Zoom, can make you oysegezoomt (a Yiddish word that more or less translates to “all Zoomed out”).  
3. We cannot incorporate physical motion into my class at all.  
4. I cannot create a technology policy that asks you, at any point, to put away every single one of your devices.

To best take advantage of the remote format while mitigating its drawbacks, I have drafted a set of technology agreements for us.

### Student Agreements (what I need you to do):

* If you have your phone with you during class, please place it in “do not disturb” mode (so only emergency calls and texts come through) and please place it face down, so that the screen lighting up does not distract you.  
    
* Please, to the extent you can, turn off notification apps on your computer like Signal, SMS, Slack, or discord.   
    
* Please close all tabs in your browser that do not relate to engaging with the course material.  
    
* During any group exercise, please have your camera turned on. This makes it easier to connect with your classmates and provides your classmates with a more inviting presentation experience than speaking to a bunch of black squares.  
    
* Beyond these two circumstances, Chelsea (the instructor) will indicate whether portions of the class should be camera-on or camera-optional on a case-by-case basis.

* Please commit to spending your mid-session break time *away from screens*, to refresh your eyeballs and brains for the second half of class.  
    
* Please ask questions in the course Slack channel (**\#intermediate-python-troy**), unless they are about your grade.   
  * For a question asked in the course Slack channel, we (the course staff) commit to a response within:  
    * 24 hours M 9A-F 5P (weekdays)  
    * 48 hours F 5P-M 9A (the weekend)  
  * For a question asked by some other means, we commit to a response time of one week (this does not apply to [requests for leniency](#important-requests-for-leniency)).  
      
* Technology use and academic honesty:  
  * *Please* collaborate on the in-class assignments.   
  * I will be specific with you about which homework assignments can be collaborative and which should not.  
  * Please feel free to ask questions about the material in the course slack channel, and if it is about code, please include the exact error message you’re seeing as well as the part of the code that you think produced it. This will not qualify as cheating.  
  * I am not banning the use of Generative AI tools in this class[^4]. I will specify, for each assignment, whether I am okay with you using them. When you use them,I expect you to speak to that use in the homework surveys.   
    

**Your participation in this course confirms your agreement to the Intermediate Python Technology Policy you have just read.** 

### Instructor Agreements (what you can expect me to do):

* Long Zoom sessions are tough. I commit to providing you with a substantial (minimum 15 minute) break in the middle of class so you have time to go to the bathroom, make yourself a coffee, eat something, or take a short walk. **I’ll remind you here of the student agreement you signed above to please commit to spending your mid-session break time *away from screens*, to refresh your eyeballs and brains for the second half of class. Look at a tree or something. Anything.**

* Part of the role of a programming education is to teach you how to use the tools available to you, from research mediums to IDEs to Generative AI tools. I commit to sharing, upon request in office hours, my methodology with you for creating or obtaining each of our course materials so that you know how I generated them, which tools I used, and how I used those tools.

* I commit to automatically applying a few common SDS accommodations to the entire class, regardless of whether a student has requested them:

  * Class recordings are uploaded to, and transcribed by, Panopto.  
  * Exams and quizzes are administered at 1.5x the amount of time suggested by the results of playtests.

  Why: I know that SDS accommodations often depend on receiving a *diagnosis*, and diagnoses are not equally easy for everyone to get. I want everyone to get their accommodation needs met in spite of this. 

## Endnotes:

[^1]:  You will receive zero bonus points in this class for tattooing the schedule on your thigh, however.  


[^2]:  You’ll note that the *letter* of [specifications grading](https://teaching.unl.edu/resources/alternative-grading/specification-grading/), as originally described in academic papers, calls for some proportion of your homeworks in the *aggregate* to be S or S+ to get various letter grades. I find this approach unnecessarily complicated, which is why I go with “S+ is full marks, S is slightly less than full marks, and so on, and the points possible in the class exceeds full credit.” I indulge my hubris to think my version accomplishes the same pedagogical goal in a less complicated way than the way described in the papers.   


[^3]:  One of the crummiest things about the reality of self-determination is that *all* decisions have consequences—including, often, the *right* decisions.  


[^4]:  With this, though, a precautionary note: in my Python Programming class, the students who use AI tools most extensively (self-reported on an end-of-quarter survey) tend to get among the worst homework grades purely based on the quality of the work. Because they’re not changing how they write their code in response to instructor feedback (since an AI tool is generating the code rather than the feedback recipient), those homework grades don’t improve over the course of the quarter. These students also routinely turn in the least ambitious final projects. They tend to end up with final grades in the B and C range. 

    My hypothesis is that this happens not because they used AI, but because a) they have yet to develop the skills needed to use AI effectively in a technical capacity and b) they’re using it to supplant an investment in the learning goals, which the grade does ultimately reflect. Conveniently in this class, the learning goals overlap *heavily* with the skills needed to excel in a technical capacity regardless of your employers’ AI policy, so I expect the correlation I’ve described to be *stronger* in Intermediate Python than in Regular Python.

    Generative AI products are a tool, like search engines or libraries. They *complement* those tools: they don’t *replace* those tools. The function of your graduate education is to learn when to use each tool, and for what, and to gain the subject matter expertise to use all these tools effectively in the future. So if you use them to try to bypass the development of the *subject matter expertise*, you’re likely to find yourself unprepared for engineering work pretty quickly.

