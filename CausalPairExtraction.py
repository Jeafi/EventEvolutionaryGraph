'''
输入中文新闻语料rawdara，输出因果事件对
'''
import os
import codecs
import json
from tqdm import tqdm

import re, jieba
import jieba.posseg as pseg
from pyltp import SentenceSplitter
import copy

class CausalityExractor():
    def __init__(self):
        pass

    '''1由果溯因配套式'''
    def ruler1(self, sentence):
        '''
        conm2:〈[之]所以,因为〉、〈[之]所以,由于〉、 <[之]所以,缘于〉
        conm2_model:<Conj>{Effect},<Conj>{Cause}
        '''
        datas = list()
        word_pairs =[['之?所以', '因为'], ['之?所以', '由于'], ['之?所以', '缘于']]
        for word in word_pairs:
            pattern = re.compile(r'\s?(%s)/[p|c]+\s(.*)(%s)/[p|c]+\s(.*)' % (word[0], word[1]))
            result = pattern.findall(sentence)
            data = dict()
            if result:
                data['tag'] = result[0][0] + '-' + result[0][2]
                data['cause'] = result[0][3]
                data['effect'] = result[0][1]
                datas.append(data)
        if datas:
            return datas[0]
        else:
            return {}
    '''2由因到果配套式'''
    def ruler2(self, sentence):
        '''
        conm1:〈因为,从而〉、〈因为,为此〉、〈既[然],所以〉、〈因为,为此〉、〈由于,为此〉、〈只有|除非,才〉、〈由于,以至[于]>、〈既[然],却>、
        〈如果,那么|则〉、<由于,从而〉、<既[然],就〉、〈既[然],因此〉、〈如果,就〉、〈只要,就〉〈因为,所以〉、 <由于,于是〉、〈因为,因此〉、
         <由于,故〉、 〈因为,以致[于]〉、〈因为,因而〉、〈由于,因此〉、<因为,于是〉、〈由于,致使〉、〈因为,致使〉、〈由于,以致[于] >
         〈因为,故〉、〈因[为],以至[于]>,〈由于,所以〉、〈因为,故而〉、〈由于,因而〉
        conm1_model:<Conj>{Cause}, <Conj>{Effect}
        '''
        datas = list()
        word_pairs =[['因为', '从而'], ['因为', '为此'], ['既然?', '所以'],
                    ['因为', '为此'], ['由于', '为此'], ['除非', '才'],
                    ['只有', '才'], ['由于', '以至于?'], ['既然?', '却'],
                    ['如果', '那么'], ['如果', '则'], ['由于', '从而'],
                    ['既然?', '就'], ['既然?', '因此'], ['如果', '就'],
                    ['只要', '就'], ['因为', '所以'], ['由于', '于是'],
                    ['因为', '因此'], ['由于', '故'], ['因为', '以致于?'],
                    ['因为', '以致'], ['因为', '因而'], ['由于', '因此'],
                    ['因为', '于是'], ['由于', '致使'], ['因为', '致使'],
                    ['由于', '以致于?'], ['因为', '故'], ['因为?', '以至于?'],
                    ['由于', '所以'], ['因为', '故而'], ['由于', '因而']]

        for word in word_pairs:
            pattern = re.compile(r'\s?(%s)/[p|c]+\s(.*)(%s)/[p|c]+\s(.*)' % (word[0], word[1]))
            result = pattern.findall(sentence)
            data = dict()
            if result:
                data['tag'] = result[0][0] + '-' + result[0][2]
                data['cause'] = result[0][1]
                data['effect'] = result[0][3]
                datas.append(data)
        if datas:
            return datas[0]
        else:
            return {}
    '''3由因到果居中式明确'''
    def ruler3(self, sentence):
        '''
        cons2:于是、所以、故、致使、以致[于]、因此、以至[于]、从而、因而
        cons2_model:{Cause},<Conj...>{Effect}
        '''

        pattern = re.compile(r'(.*)[,，]+.*(于是|所以|故|致使|以致于?|因此|以至于?|从而|因而)/[p|c]+\s(.*)')
        result = pattern.findall(sentence)
        data = dict()
        if result:
            data['tag'] = result[0][1]
            data['cause'] = result[0][0]
            data['effect'] = result[0][2]
        return data
    '''4由因到果居中式精确'''
    def ruler4(self, sentence):
        '''
        verb1:牵动、导向、使动、导致、勾起、引入、指引、使、予以、产生、促成、造成、引导、造就、促使、酿成、
            引发、渗透、促进、引起、诱导、引来、促发、引致、诱发、推进、诱致、推动、招致、影响、致使、滋生、归于、
            作用、使得、决定、攸关、令人、引出、浸染、带来、挟带、触发、关系、渗入、诱惑、波及、诱使
        verb1_model:{Cause},<Verb|Adverb...>{Effect}
        '''
        pattern = re.compile(r'(.*)\s+(牵动|已致|导向|使动|导致|勾起|引入|指引|使|予以|产生|促成|造成|引导|造就|促使|酿成|引发|渗透|促进|引起|诱导|引来|促发|引致|诱发|推进|诱致|推动|招致|影响|致使|滋生|归于|作用|使得|决定|攸关|令人|引出|浸染|带来|挟带|触发|关系|渗入|诱惑|波及|诱使)/[d|v]+\s(.*)')
        result = pattern.findall(sentence)
        data = dict()
        if result:
            data['tag'] = result[0][1]
            data['cause'] = result[0][0]
            data['effect'] = result[0][2]
        return data
    '''5由因到果前端式模糊'''
    def ruler5(self, sentence):
        '''
        prep:为了、依据、为、按照、因[为]、按、依赖、照、比、凭借、由于
        prep_model:<Prep...>{Cause},{Effect}
        '''
        pattern = re.compile(r'\s?(为了|依据|按照|因为|因|按|依赖|凭借|由于)/[p|c]+\s(.*)[,，]+(.*)')
        result = pattern.findall(sentence)
        data = dict()
        if result:
            data['tag'] = result[0][0]
            data['cause'] = result[0][1]
            data['effect'] = result[0][2]

        return data

    '''6由因到果居中式模糊'''
    def ruler6(self, sentence):
        '''
        adverb:以免、以便、为此、才
        adverb_model:{Cause},<Verb|Adverb...>{Effect}
        '''
        pattern = re.compile(r'(.*)(以免|以便|为此|才)\s(.*)')
        result = pattern.findall(sentence)
        data = dict()
        if result:
            data['tag'] = result[0][1]
            data['cause'] = result[0][0]
            data['effect'] = result[0][2]
        return data

    '''7由因到果前端式精确'''
    def ruler7(self, sentence):
        '''
        cons1:既[然]、因[为]、如果、由于、只要
        cons1_model:<Conj...>{Cause},{Effect}
        '''
        pattern = re.compile(r'\s?(既然?|因|因为|如果|由于|只要)/[p|c]+\s(.*)[,，]+(.*)')
        result = pattern.findall(sentence)
        data = dict()
        if result:
            data['tag'] = result[0][0]
            data['cause'] = result[0][1]
            data['effect'] = result[0][2]
        return data
    '''8由果溯因居中式模糊'''
    def ruler8(self, sentence):
        '''
        3
        verb2:根源于、取决、来源于、出于、取决于、缘于、在于、出自、起源于、来自、发源于、发自、源于、根源于、立足[于]
        verb2_model:{Effect}<Prep...>{Cause}
        '''

        pattern = re.compile(r'(.*)(根源于|取决|来源于|出于|取决于|缘于|在于|出自|起源于|来自|发源于|发自|源于|根源于|立足|立足于)/[p|c]+\s(.*)')
        result = pattern.findall(sentence)
        data = dict()
        if result:
            data['tag'] = result[0][1]
            data['cause'] = result[0][2]
            data['effect'] = result[0][0]
        return data
    '''9由果溯因居端式精确'''
    def ruler9(self, sentence):
        '''
        cons3:因为、由于
        cons3_model:{Effect}<Conj...>{Cause}
        '''
        pattern = re.compile(r'(.*)是?\s(因为|由于)/[p|c]+\s(.*)')
        result = pattern.findall(sentence)
        data = dict()
        if result:
            data['tag'] = result[0][1]
            data['cause'] = result[0][2]
            data['effect'] = result[0][0]

        return data

    '''抽取主函数'''
    def extract_triples(self, sentence):
        infos = list()
      #  print(sentence)
        if self.ruler1(sentence):
            infos.append(self.ruler1(sentence))
        elif self.ruler2(sentence):
            infos.append(self.ruler2(sentence))
        elif self.ruler3(sentence):
            infos.append(self.ruler3(sentence))
        elif self.ruler4(sentence):
            infos.append(self.ruler4(sentence))
        elif self.ruler5(sentence):
            infos.append(self.ruler5(sentence))
        elif self.ruler6(sentence):
            infos.append(self.ruler6(sentence))
        elif self.ruler7(sentence):
            infos.append(self.ruler7(sentence))
        elif self.ruler8(sentence):
            infos.append(self.ruler8(sentence))
        elif self.ruler9(sentence):
            infos.append(self.ruler9(sentence))

        return infos

    '''抽取主控函数'''
    def extract_main(self, content):
        sentences = self.process_content(content)
        datas = list()
        for sentence in sentences:
            subsents = self.fined_sentence(sentence)
            subsents.append(sentence)
            for sent in subsents:
                sent = ' '.join([word.word + '/' + word.flag for word in pseg.cut(sent)])
                result = self.extract_triples(sent)
                if result:
                    for data in result:
                        if data['tag'] and data['cause'] and data['effect']:
                            datas.append(data)
        return datas

    '''文章分句处理'''
    def process_content(self, content):
        return [sentence for sentence in SentenceSplitter.split(content) if sentence]

    '''切分最小句'''
    def fined_sentence(self, sentence):
        return re.split(r'[？！，；]', sentence)

def extract_to_json():
    path = r'rawdata'
    writepath = r'cepairs'
    files = os.listdir(path)
    fw = open('causeeffect.json', 'w', encoding='utf-8')
    # 
    # FilterFile = open('WordsDic/filter.txt', 'r', encoding='utf8')
    # Filter = []
    # TiggerFile = open('WordsDic/tigger.txt', 'r', encoding='utf8')
    # tigger = []
    # for fWord in FilterFile.readlines():
    #      Filter.append(fWord.strip())
    # # for tig in TiggerFile.readlines():
    # #     tigger.append(tig.strip())
    # FilterFile.close()
    # TiggerFile.close()
    # print(Filter)
    # print(tigger)
    for file in tqdm(files):
        if not os.path.isdir(file):
            f = open(os.path.join(path, file), 'r', encoding='utf-8')
            lines = f.readlines()
            f.close()
            fw = open(os.path.join(writepath, file.replace('.txt','_ce.json')), 'w', encoding='utf8')
            titles = []
            extractor = CausalityExractor()
            serial0 = int(''.join(file.split('.')[0].split('-')))*100000
            serial = serial0
            for line in tqdm(lines):
                # flag = False
                # for fWord in Filter:
                #     if line.find(fWord) != -1:
                #         flag = True 
                #         break
                # if flag == False: #如果不包含需要过滤的词
                    # newSp = line.split('#')
                    # serial = newSp[0]
                    # content = newSp[1] +'。'+newSp[2]
                dr = re.compile(r'<[^>]+>',re.S)
                title = dr.sub('',json.loads(line).get('title'))
                content = dr.sub('',json.loads(line).get('content'))
                data = extractor.extract_main(title+'。'+content)
                lastpair = dict()
                for d in data:
                    if lastpair == d:
                        continue
                    else:
                        d['serial'] = serial
                        serial +=1
                        d['title'] = title
                        d['content'] = content
                        json.dump(d, fw, ensure_ascii=False)
                        fw.write('\n')
            # s = []
            # lastone = dict()
            # for i, title in tqdm(enumerate(titles)):
            #     for d in data:
            #         if lastone == d:
            #             continue
            #         else:
            #             temp_dict = {}
            #             temp_dict['title'] = json.loadline[i]
            #             s.append()
            #         lastone = d
            # serial = int(''.join(file.split('.')[0].split('-')))*100000
            # serial0 = serial
            # print(len(s),len(lines))
            # for i, data in enumerate(s):
            #     data['serial'] = serial
            #     data['title'] = json.loads(lines[i]).get('title')
            #     json.dump(data, fw, ensure_ascii=False)
            #     serial = serial+1
            #     fw.write('\n')
            fw.close()
            print(serial-serial0)

def extract_for_srl():
    '''
    为srl生成训练数据
    '''
    path = r'rawdata'
    writepath = r'cepairs'
    files = os.listdir(path)
    fw = open('srl.txt', 'w', encoding='utf-8')
    # 
    # FilterFile = open('WordsDic/filter.txt', 'r', encoding='utf8')
    # Filter = []
    # TiggerFile = open('WordsDic/tigger.txt', 'r', encoding='utf8')
    # tigger = []
    # for fWord in FilterFile.readlines():
    #      Filter.append(fWord.strip())
    # # for tig in TiggerFile.readlines():
    # #     tigger.append(tig.strip())
    # FilterFile.close()
    # TiggerFile.close()
    # print(Filter)
    # print(tigger)
    s = set()
    for file in tqdm(files):
        if not os.path.isdir(file):
            f = open(os.path.join(path, file), 'r', encoding='utf-8')
            lines = f.readlines()
            f.close()
            # fw = open(os.path.join(writepath, file), 'w', encoding='utf8')
            titles = []
            extractor = CausalityExractor()
            for line in tqdm(lines):
                # flag = False
                # for fWord in Filter:
                #     if line.find(fWord) != -1:
                #         flag = True 
                #         break
                # if flag == False: #如果不包含需要过滤的词
                    # newSp = line.split('#')
                    # serial = newSp[0]
                    # content = newSp[1] +'。'+newSp[2]
                title = json.loads(line).get('title')
                content = json.loads(line).get('content')
                titles.append(title+'。 '+content)
            lastone = dict()
            for title in tqdm(titles):
                dr = re.compile(r'<[^>]+>',re.S)
                title = dr.sub('',title)
                s = ['O']*len(title)
                data = extractor.extract_main(title)
                # sentences = title.split('。')
                # for s in sentences:
                indexlist = []
                for d in data:
                    try:
                        for tag in d['tag'].split('-'):
                            indexlist = [i.start() for i in re.finditer(tag, title)]
                            for i in indexlist:
                                s[i] = 'B-TIG'
                                for j in range(1,len(tag)):
                                    s[i+j] = 'I-TIG'
                            # print(indexlist)
                            # print(s)
                        for cause in d['cause']:
                            cause = ''.join([word.split('/')[0] for word in d['cause'].split(' ') if word.split('/')[0]])
                            try:
                                indexlist = [i.start() for i in re.finditer(cause, title)]
                            except:
                                pass
                            for i in indexlist:
                                s[i] = 'B-CAUSE'
                                for j in range(1,len(cause)):
                                    s[i+j] = 'I-CAUSE'
                            for i in indexlist:
                                s[i] = 'B-CAUSE'
                                for j in range(1,len(cause)):
                                    s[i+j] = 'I-CAUSE'
                            # print(indexlist)
                            # print(s)
                        for effect in d['effect']:
                            effect = ''.join([word.split('/')[0] for word in d['effect'].split(' ') if word.split('/')[0]])
                            try:
                                indexlist = [i.start() for i in re.finditer(effect, title)]
                            except:
                                pass
                                # print(d)
                                # print(title)
                            for i in indexlist:
                                s[i] = 'B-EFFECT'
                                for j in range(1,len(effect)):
                                    s[i+j] = 'I-EFFECT'
                            # print(indexlist)
                            # print(s)
                    except:
                        pass
                for i in range(len(title)):
                    if title[i] == ' ':
                        fw.write('\n')
                    else:
                        fw.write(title[i]+' '+s[i]+'\n')
                fw.write('\n')

if __name__ == "__main__":
    extract_to_json()

    # extractor = CausalityExractor()
    # data = extractor.extract_main("之所以爱你是因为我是aaa。因为我是aaa所以我用你。")
    # print(data)
    # print(len(data))
    # extract_for_srl()