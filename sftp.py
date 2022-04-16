import pysftp
import time

host = '203.250.148.52' # 호스트명만 입력. sftp:// 는 필요하지 않다.
port = 20514 # int값으로 sftp서버의 포트 번호를 입력
username = 'yujin' # 서버 유저명
password = 'aisl1234!' # 유저 비밀번호

hostkeys = None

# 서버에 저장되어 있는 모든 호스트키 정보를 불러오는 코드
cnopts = pysftp.CnOpts()

# 접속을 시도하는 호스트에 대한 호스트키 정보가 존재하는지 확인
# 존재하지 않으면 cnopts.hostkeys를 None으로 설정해줌으로써 첫 접속을 가능하게 함
if cnopts.hostkeys.lookup(host) == None:
    print("Hostkey for " + host + " doesn't exist")
    hostkeys = cnopts.hostkeys # 혹시 모르니 다른 호스트키 정보들 백업
    cnopts.hostkeys = None

# 첫 접속이 성공하면, 호스트에 대한 호스트키 정보를 서버에 저장.
# 두번째 접속부터는 호스트키를 확인하며 접속하게 됨.

# sftp 접속을 실행
with pysftp.Connection(
                        host,
                        port = port,
                        username = username,
                        password = password,
                        cnopts = cnopts) as sftp:
    
    # 접속이 완료된 후 이 부분이 호스트키를 저장하는 부분
    # 처음 접속 할 때만 실행되는 코드
    if hostkeys != None:
        print("New Host. Caching hostkey for " + host)
        hostkeys.add(host, sftp.remote_server_key.get_name(), sftp.remote_server_key) # 호스트와 호스트키를 추가
        hostkeys.save(pysftp.helpers.known_hosts()) # 새로운 호스트 정보 저장

    # 폴더에 있는 모든 파일들을 한거번에 업로드 하고 싶을 땐 'put_d' 를 사용
    # 예) sftp.put_d('업로드 할 파일들이 있는 폴더 경로', '/')
    
    # 여러 파일들을 개별로 업로드 하고 싶을 땐 'put'을 여러번 사용
    # 예) sftp.put('파일1 경로')
    # 예) sftp.put('파일2 경로')
    
    # sftp서버에 있는 파일과 폴더들을 보고 싶을 땐 아래 함수 실행
    print(sftp.listdir('/home/yujin'))
    start = time.time()
    sftp.get('/home/yujin/beverage-img-cf.hdf5') #서버에서 가져올 때, get // 모델: beverage-img-cf.hdf5 용량: 97.1MB --> 17초
    print(time.time()-start)
    sftp.put('C:\\Users\\YujinKim\\Dropbox\\YujinKim\\랩실\\02. 협업학습\\socialLerning\\img_5doong.jpg') #로컬에서 서버로 보낼 때, put
    
    # 모든 작업이 끝나면 접속 종료
    sftp.close()