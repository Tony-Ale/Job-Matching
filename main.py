"""This code collects user ID, (in this case numbers based from the user.csv file)
Generates a list of tags from user interests and courses the user scored 50 and above
Then uses cosine_similarity (natural language processing) to match the tags to the list of job descriptions,
thereafter it outputs a sorted list based on the user country; containing the
first twenty jobs with the highest similarity.Were jobs in the users country comes first"""

import csv
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import operator
import os.path

file = open("3.users.csv")
file2 = open("7.skills.csv")
file3 = open("5.jobs.csv", encoding="utf8")
file4 = open("2.tags.csv")
file5 = open("1.countries.csv")

csvreader = csv.reader(file)
csvreader2 = csv.reader(file2)
csvreader3 = csv.reader(file3)
csvreader4 = csv.reader(file4)
csvreader5 = csv.reader(file5)

header = next(csvreader)
header2 = next(csvreader2)
header3 = next(csvreader3)
header4 = next(csvreader4)
header5 = next(csvreader5)

# reading user.csv
rows = []
for row in csvreader:
        rows.append(row)

# reading skills.csv
rows2 = []
for roww in csvreader2:
    rows2.append(roww)
# reading jobs.csv
rows3 = []
for rowww in csvreader3:
    rows3.append(rowww)
# reading tags.csv
rows4 =[]
for rowwww in csvreader4:
    rows4.append(rowwww)
# reading countries.csv
countryLists = []
for countries in csvreader5:
    countryLists.append(countries)
# extracting singly the id and country id column from user.csv
idCol = []
countryId = []
for col in rows:
    idCol.append(col[0])
    countryId.append(col[2])
# extracting singly; tags, score and userId column from skills.csv
tagsCol = []
scoreCol = []
skillsUserId = []
for column in rows2:
    tagsCol.append(column[2])
    scoreCol.append(column[3])
    skillsUserId.append(column[1])
# extracting tags to column from jobs.csv
tagsJobs = []
for column2 in rows3:
    tagsJobs.append(column2[7])
# Original tags to column from tags.csv
tags = []
for columnId in rows4:
    tags.append(columnId[0])
# extracting countries Id from countries.csv
mainCountryId = []
for locationID in countryLists:
    mainCountryId.append(locationID[0])

# Asking for user input and getting user interest
while True:
    try:
        userID = input("input user ID number: ")
        index = idCol.index(userID)
        break
    except ValueError:
        print("pls input a valid userID")
        continue

userInterests = rows[index][4]
userCountryIndex = mainCountryId.index(countryId[index])
userCountry = countryLists[userCountryIndex][1]
print("user country is ", userCountry)

# breaking down user interest in bits
userInterestsTag = list(userInterests.split("/"))

# separating skills per user tags
skillsTagsPerUser = []
count = 0
for elements in skillsUserId:
    if elements == userID:
        elementIndex = skillsUserId.index(elements, count)
        skillsTagsPerUser.append(tagsCol[elementIndex]) # just decided to have a separate list for skills tags
        if scoreCol[elementIndex].strip() >= "50": # ensures that any course the user scored less than 50 should not be added to skills tags
            userInterestsTag.append(tagsCol[elementIndex])
        count = elementIndex+1

# converting user interests tag to strings
userTagList = []
for element in userInterestsTag:
    value = tags.index(element)
    userTagList.append(rows4[value][1])

userTagString = " ".join(map(str, userTagList))
print("Generated user tags are: ", userTagString)

# matching job descriptions to generated user tags.
result = {}
for j in range(len(rows3)):
    corpus = [rows3[j][3], userTagString]
    vectorizer = TfidfVectorizer()
    trsfm = vectorizer.fit_transform(corpus)
    #pd.DataFrame(trsfm.toarray(), columns=vectorizer.get_feature_names_out(), index=['jobDescription', 'userTags'])
    similarity = (cosine_similarity(trsfm[0:1], trsfm))
    result[j] =similarity.item(1)
sortedResult = dict( sorted(result.items(), key=operator.itemgetter(1),reverse=True))

# converting sorted results to strings, that is the jobs
jobTitle = []
company = []
jobDesc = []
sortedResultIndex = []
jobsID = []
for eachValue in sortedResult.keys():
    jobTitle.append(rows3[eachValue][1])
    company.append(rows3[eachValue][2])
    jobDesc.append(rows3[eachValue][3])
    jobsID.append(rows3[eachValue][0])
    sortedResultIndex.append(eachValue)

# extracting the first twenty jobs and sorting them out by country
firstTwentyJobTitle = []
firstTwentyCompany = []
firstTwentyJobDesc = []
firstTwentyJobID = []
for jobs in range(20):
    firstTwentyJobTitle.append(jobTitle[jobs])
    firstTwentyCompany.append(company[jobs])
    firstTwentyJobDesc.append(jobDesc[jobs])
    firstTwentyJobID.append(jobsID[jobs])

# matching user country to job location and sorting it out
sortedJobs = {}
for j in range(20):
    corpus = [rows3[sortedResultIndex[j]][5], userCountry]
    vectorizer = TfidfVectorizer()
    trsfm = vectorizer.fit_transform(corpus)
    #pd.DataFrame(trsfm.toarray(), columns=vectorizer.get_feature_names_out(), index=['sortedResult', 'userCountry'])
    similarity = (cosine_similarity(trsfm[0:1], trsfm))
    sortedJobs[j] =similarity.item(1)
sortedJobsBasedOnCountry = dict( sorted(sortedJobs.items(), key=operator.itemgetter(1),reverse=True))
#print(sortedJobsBasedOnCountry)

# converting sorted results to strings
sortedJobTitle = []
sortedCompany = []
sortedIndex = []
sortedID = []
for eachValue in sortedJobsBasedOnCountry.keys():
    sortedJobTitle.append(firstTwentyJobTitle[eachValue])
    sortedCompany.append(firstTwentyCompany[eachValue])
    sortedID.append(firstTwentyJobID[eachValue])
    sortedIndex.append(eachValue)


for eachCompany in range(len(sortedCompany)):
    print(eachCompany+1, ". ", sortedCompany[eachCompany], ": ", sortedJobTitle[eachCompany])

# writing to csv file
header = ["userID"]
sortedID.insert(0,userID)
for l in range(20):
    header.append("jobID")
file_exists = os.path.exists("jobprediction.csv")
resultFile = open("jobprediction.csv", "a", encoding='UTF8', newline='')
writer = csv.writer(resultFile)
if not file_exists:
    writer.writerow(header)
writer.writerow(sortedID)
#print(sortedID)
while True:
    try:
        print("\nif you want to end the program type the number zero (0)")
        serialNumber = int(input("pls input serial number of listed jobs to see description: "))
        if serialNumber == 0:
            print("Thanks for using this program")
            break
        elif serialNumber > 20 or serialNumber < 0:
            print("pls input a valid number")
            continue
        elif serialNumber <= 20:
            indexes = sortedIndex[serialNumber - 1]
            print(firstTwentyJobDesc[indexes])
    except ValueError:
        print("pls input integer only...")
        continue








