import os
import nltk.data
from nltk.tokenize import word_tokenize
import regex
from itertools import groupby
from bs4 import BeautifulSoup



SYMBOLS_TO_STRIP="/'*+,-.=:"

REPLACE_PATTENRS = [("((накл)|(накл.)|(ах)|(апв)|(ао)|(холдинг)|(фи)|(заявка)|(г)|(афи)|(апт)|(заявке)|(ба)|(тн)|(тмрс)|(сиа)|(роста)|(пфи)|(пульс)|(протек)|(протека))(?=№)", "\\1 "),
	("№(?=[^\s])", "\\0 "),
	("((цм)|(штрих)|(штрихкод)|(ap)|(цене)|(ш\/код)|(уткз)|(ш-код)|(р)|(сл)|(сери)|(се)|(ом)|(уткз)|(сбт)|(сумма)|(сп)|(сумму)|(торг)|(с)|(от)|(по)|(в)|(ди)|(ая)|(ар)|(бо)|(блп)|(бд)(верси)(код)|(вп)|(дм)|(еоп)|(екр)|(ебо)|(ебд)|(еая)|(ебт)|(ев)|(екп)|(оп)|(отчету)|(отчете)|(см)|(св)|(рн-иж)|(рн-йо)|(сери)|(сбд)|(поз)|(сб)|(сди)|(сер)|(смvs)|(серия)|(кзн)|(фи)|(ф)|(ут)|(у)|(тмрс)|(тмр)|(пфи)|(пф)|(нпр)|(ка)|(каб)|(к)|(итрс)|(имрс)|(импр)|(идрс)|(выручка)|(бп)|(бс)|(бн)|(бл)|(бж)|(аь)|(аш)|(ач)|(ац)|(ах)|(аф)|(ат)|(апв)|(ап)|(ао)|(ай)|(аи)|(ае)|(адрес)|(аг)|(ав)|(аб)|(а)|(пр))(?=\d)", "\\1 "),
	("г\.(?=\p{Cyrillic})", "г. "),
	("\d(?=\p{Cyrillic})", "\\0 "),
	("(?<=(штрих)|(дс)|(цена)|(торг)|(терм)|(тел)|(д)|(ш)|(шт)|(стм)|(ср\.ст)|(чеков)|(чека)|(ценам)|(цене)|(наличными)|(алмаг)|(бад)|(бижи)|(безнал)|(наличные)|(нал)|(регистру)|(рбн)|(нп1)|(одна)|(ошибку)|(отчету)|(отчете)|(ошибка)|(разница)|(продаж)|(программе)|(покуп)|(поз)|(сч)|(сумма)|(сумму)|(сегодня)|(скидка)|(серия)|(скид)|(скату)|(сироп)|(код)|(ут)|(уи)|(у)|(р)|(пфи)|(каб)|(к)|(выручка)|(выплата)|(б\/нал)|(ах)|(апв)|(ап)|(аоц)|(аос)|(ао)|(адрес)|(продаж)|(программе))[:=\-\.](?=\d)", " \\1 "),
	("(?<=уп),(?=\S)", "\\0 "),
	("(?<=(уп)|(юр)|(ул)|(см)|(эл)|(с)|(раб)|(тыс)|(шт)|(шк)|(фл)|(т)|(тех)|(чек)|(таб)|(т\.к)|(гл)|(амп)|(т\.е)|(руб)|(гр)|(дет)|(день)|(дез)|(зуб)|(зал)|(др)|(зав)|(есть)|(здравствуйте)|(розн)|(нет)|(налич)|(нал)|(накл)|(раз)|(покуп)|(опт)|(поз)|(пож)|(пож-та)|(пл)|(пр)|(пожалуйста))\.(?=\S)", "\\1 "),
	("прошу(?=\p{Cyrillic})", "\\0 "),
	("д/", "для "),
	("б/", "без "),
	("д.д.", "ДД "),
	("д.б.", "должно быть "),
	("(?<=\d)\.(?=\p{Cyrillic})", " \\0 "),
	("(?<=ошибка)=(?=-{0,1}\d)", " \\0 "),
	("(?<=\p{Cyrillic}\p{Cyrillic}\p{Cyrillic}\p{Cyrillic})[\.\-](?=\p{Cyrillic}{3,})", " \\0 ")]

ABBREVIATIONS = [("г", "город"),
				 ("бух", "бухгалтерия"),
				 ("бух-е", "бухгалтерские"),
				 ("бух-ия", "бухгалтерия"),
				 ("бух-й", "бухгалтерский"),
				 ("бух-р", "бухгалтер"),
				 ("бух-ра", "бухгалтера"),
				 ("бух-рия", "бухгалтерия"),
				 ("бухг", "бухгалтерский"),
				 ("бух-ю", "бухгалтерию"),
				 ("бум", "бумажная"),
				 ("банк", "банковский"),
				 ("б/н", "б/нал"),
				 ("уп", "упаковка"),
				 ("ул", "улица"),
				 ("тыс", "тысяч"),
				 ("тех", "технический"),
				 ("пож", "пожалуйста"),
				 ("пож-та", "пожалуйста"),
				 ("пж", "пожалуйста"),
				 ("пжл", "пожалуйста"),
				 ("пж-ста", "пожалуйста"),
				 ("пж-та", "пожалуйста"),
				 ("пжс", "пожалуйста")]

URL_PATTERN = regex.compile("\.com|\.ru|\.net")
LOCALPATH_PATTERN = regex.compile("home\/")
FLOAT_NUMBER_PATTERN = regex.compile("^\d+[,\.]\d+$")

PRICE_PATTERN = regex.compile("\dкоп|\dр|\dт.р")
WEIGHT_PATTERN = regex.compile("\dмг|\dг|\dмкг")
VOLUME_PATTERN = regex.compile("\dмл")
IP_PATTERN = regex.compile("\d{2,3}[,\.\/]\d{1,3}[,\.\/]\d{1,3}[,\.\/]\d{1,3}")
DATE_PATTERN = regex.compile("\d{1,2},\d{1,2},\d{2,4}|\d{1,2}\.\d{1,2}\.\d{2,4}|\d{1,2}\/\d{1,2}\/\d{2,4}")
DATE2_PATTERN = regex.compile("^0\d[,\.]\d\d|^\d\d[,\.]\d{1,2}[,\.]\d{1,4}")
TIME_PATTERN = regex.compile("\d{1,2}:\d{1,2}\d{1,2}|\d{1,2}:\d{1,2}")
NUMBER_PATTERN = regex.compile("^№|фк\d+|фи\d+|ф\d+|ут\d+|ут-\d+|тмрс\d+|р-\d+|пфи\d+|пфи-\d+|пф\d+|нпр\d+")
DIMS_PATTERN = regex.compile("\d+[\*x]\d+")
LIST_OF_BIG_INTS_PATTERN = regex.compile("\d{5,}(,\d{5,})+")
BIG_INT_PATTERN = regex.compile("\d{5,}")
ENGLISH_PATTERN = regex.compile("[a-zA-Z]{4,}")


CODE1_PATTERN = regex.compile("\d+_\d+")
CODE2_PATTERN = regex.compile("\d+\/\d+")
CODE3_PATTERN = regex.compile("\d+-\d+")

def split_sentence_to_words(sentence):
	sentence = sentence_cleaning(sentence)
	return split_cleaned_sentence_to_words(sentence)

def split_cleaned_sentence_to_words(cleaned_sentence):
	words = word_tokenize(cleaned_sentence)
	words = [word_cleaning(word) for word in words]
	#words = [word for word in words if not any(ch.isdigit() for ch in word)]
	words = [word for word in words if len(word) > 0]
	return words

def sentence_cleaning(sentence):
	soup = BeautifulSoup(sentence, "lxml")
	for table in soup.find_all('table'):
		if len(table.findAll('tr')) > 5:
			table.replaceWith('')
	sentence = " ".join(soup.strings)
	sentence = sentence.lower()
	while True:
		l = len(sentence)
		#for replace_pattern in REPLACE_PATTENRS:
		#	sentence = regex.sub(replace_pattern[0], replace_pattern[1], sentence)
		if len(sentence) == l:
			break
	sentence = sentence.strip(" ")
	#sentence = regex.sub("\d", "", sentence)
	#sentence = regex.sub("[a-z]", "", sentence)
	#sentence = regex.sub("[^\p{Cyrillic}]"," ", sentence)
	return sentence

def word_cleaning_for_digit(word):
	word = regex.sub('\d', '0', word)
	return word

def word_cleaning(word):
	while len(word) > 1:
		l = len(word)
		for ch in SYMBOLS_TO_STRIP:
			word = word.strip(ch)
		if len(word) == l:
			break
	for abbr in ABBREVIATIONS:
		if word == abbr[0]:
			word = abbr[1]
	if all(ch.isdigit() for ch in word):
		return '%digits%'
	if FLOAT_NUMBER_PATTERN.search(word) is not None:
		return '%float%'
	if NUMBER_PATTERN.search(word) is not None:
		return '%number%'
	if URL_PATTERN.search(word) is not None:
		return '%url%'
	if LOCALPATH_PATTERN.search(word) is not None:
		return '%localpath%'
	if PRICE_PATTERN.search(word) is not None:
		return '%price%'
	if WEIGHT_PATTERN.search(word) is not None:
		return '%weight%'
	if IP_PATTERN.search(word) is not None:
		return '%ip%'
	if DATE_PATTERN.search(word) is not None:
		return '%date%'
	if DATE2_PATTERN.search(word) is not None:
		return '%date%'
	if TIME_PATTERN.search(word) is not None:
		return '%time%'
	if CODE1_PATTERN.search(word) is not None:
		return '%code1%'
	if CODE2_PATTERN.search(word) is not None:
		return '%code2%'
	if CODE3_PATTERN.search(word) is not None:
		return '%code3%'
	if DIMS_PATTERN.search(word) is not None:
		return '%dims%'
	if LIST_OF_BIG_INTS_PATTERN.search(word) is not None:
		return '%listofbigints%'
	if BIG_INT_PATTERN.search(word) is not None:
		return '%bigint%'
	if ENGLISH_PATTERN.search(word) is not None:
		return '%english%'


	word = ''.join(ch for ch, _ in groupby(word))
	return word