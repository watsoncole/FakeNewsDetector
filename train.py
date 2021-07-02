import os
import numpy as np
import requests
from bs4 import BeautifulSoup as banger
from collections import Counter
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn.metrics import confusion_matrix

BLACKLISTED = ['the','be','and','of','a','in','to','it','or','an','its','that','is','for','was','on']

def make_Dictionary():
    all_words = []
    paths = ['./Fake/','./Real/']
    for pizza in paths:
         fs = os.listdir(pizza)
         for i in fs:
            f = open(pizza+i)
            fc = f.read()
            f.close()
            words = fc.split();
            for word in words:
                if word not in BLACKLISTED:
                    all_words.append(word)

    dictionary = Counter(all_words)
    # removal of nonwords
    list_to_remove = dictionary.keys()
    for item in list_to_remove:
       if item.isalpha() == False:
           del dictionary[item]
       elif len(item) == 1:
           del dictionary[item]
    dictionary = dictionary.most_common(3000)
    return dictionary

def extract_features():
    pathFake = './Fake/'
    pathReal = './Real/'
    fs = os.listdir(pathReal)
    fx = os.listdir(pathFake)
    features_matrix = np.zeros((len(fx)+len(fs),3000))
    docID = 0
    for i in (fs + fx):
       try:
           f = open(pathFake + i)
       except:
           f = open(pathReal + i)
       q = f.read()
       f.close()
       #recapture q
       words = q.split()
       for word in words:
           for i,d in enumerate(dictionary):
               if d[0] == word:
                   wordID = i
                   features_matrix[docID,wordID] = words.count(word)
       docID = docID + 1
    return features_matrix



def GetWebsiteFromUrl(url):
    html = requests.get(url)
    if(html.status_code >= 300):
        raise Exception("URL Returned an invalid response (", html.status_code, ").")
        return ""
    style = banger(html.text, 'html.parser')
    # kill all script and style elements
    for script in style(["script", "style"]):
        script.extract()
    text = style.get_text()
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = ' '.join(chunk for chunk in chunks if chunk)
    #creating form for neural network to test
    words = text.split()
    web_matrix = np.zeros(3000)
    for word in words:
        for i,d in enumerate(dictionary):
            if d[0] == word:
                wordID = i
                web_matrix[wordID] = words.count(word)
    return web_matrix.reshape(1, -1)


train_dir = 'train-data'
dictionary = make_Dictionary()
train_labels = np.zeros(326)
train_labels[198:325] = 1
train_matrix = extract_features()
#seperate sh** stuff here

#0 is fake 1 is real
#multinomial Naive Bayes
model1 = MultinomialNB()
#Linear Support Vector Classifier
model2 = LinearSVC()
model1.fit(train_matrix,train_labels)
model2.fit(train_matrix,train_labels)


#test time : )
test_matrix = extract_features()
test_labels = np.zeros(326)
test_labels[198:325] = 1
result1 = model1.predict(test_matrix)
result2 = model2.predict(test_matrix)

print(confusion_matrix(test_labels,result1))
print(confusion_matrix(test_labels,result2))


def main():
	print("Neural Networks trained ready for input!\n")
	submit = ' ';
	while(submit!='quit'):
		submit = raw_input("Enter url of website to check if it is fake/bias news or type quit to quit: ")
		print("Multinomial Naive Bayes result for webpage:")
		try:
			m1 = model1.predict(GetWebsiteFromUrl(submit))
			if(m1==[0.]):
				print("Fake/Biased")
			else:
				print("Real/Unbiased")
			print("Linear Suport Vector Classifier result for webpage:")
			m2 = model2.predict(GetWebsiteFromUrl(submit))
			if(m2==[0.]):
				print("Fake/Biased")
			else:
				print("Real/Unbiased")
			print("---------------------------------------------------")
		except e:
                	print(e)

main()
