import React, { useEffect, useReducer } from 'react';
import { Image, Pressable, ScrollView, StyleSheet, Text, View } from 'react-native';
import { catalog, hostId, initialDemo, selectDemoView, updateDemo } from './linkingDemo';

const art = {
  winter: require('./assets/winter-window.png'),
  host: require('./assets/host.png'),
  guest: require('./assets/friend.png'),
  goldHeart: require('./assets/heart-gold.png'),
  clayHeart: require('./assets/heart-clay.png'),
};
const photos = {
  cafe: require('./assets/cafe.jpg'),
  friend: require('./assets/friend.jpg'),
  snow: require('./assets/snow.jpg'),
  tea: require('./assets/tea.jpg'),
};
const color = {
  paper: '#FAF3E8', white: '#FFF9F0', ink: '#233345', muted: '#4C5D6B',
  line: '#D8CFC3', navy: '#253953', night: '#182B43', gold: '#F4C97F',
  blue: '#BCD0E4', soft: '#E8EAF1', sage: '#DDE5D7', sageInk: '#354D42',
};
const nextDemoStep = {
  waiting: ['friendSubmitted', 1850],
  analyzing: ['analysisReady', 1250],
  visibilityWait: ['friendApproved', 1300],
  confirmationWait: ['friendConfirmed', 1700],
};

function Pixel({ source, width, height = width }) {
  return <Image source={source} style={{ width, height }} accessible={false} />;
}

function Button({ label, onPress, secondary = false, disabled = false, flex = false }) {
  return (
    <Pressable accessibilityRole="button" accessibilityState={{ disabled }} onPress={onPress}
      disabled={disabled} style={({ pressed }) => [
        styles.button, secondary && styles.secondary, disabled && styles.disabled,
        flex && styles.flex, pressed && styles.pressed,
      ]}>
      <Text style={[styles.buttonLabel, secondary && styles.secondaryLabel]}>{label}</Text>
    </Pressable>
  );
}

function Intro({ title, body, dark = false }) {
  return (
    <View style={styles.intro}>
      <Text accessibilityRole="header" style={[styles.heading, dark && styles.paperText]}>{title}</Text>
      <Text style={[styles.description, dark && styles.blueText]}>{body}</Text>
    </View>
  );
}

function Shell({ children, action, title = '잇기', onBack, onArchive, dark = false }) {
  const home = title === '올해 우리는';
  return (
    <View style={[styles.outer, dark && styles.darkOuter]}>
      <View style={[styles.screen, dark && styles.darkScreen]}>
        <View style={[styles.top, home && styles.homeTop]}>
          {home ? <Pixel source={art.clayHeart} width={21} height={18} /> : (
            <Pressable accessibilityRole="button" accessibilityLabel="홈으로 돌아가기" onPress={onBack} style={styles.back}>
              <Text style={[styles.backText, dark && styles.paperText]}>{'‹'}</Text>
            </Pressable>
          )}
          <Text style={[styles.navTitle, home && styles.homeName, dark && styles.paperText]}>{title}</Text>
          {home ? (
            <Pressable accessibilityRole="button" onPress={onArchive} style={styles.archiveButton}>
              <Text style={styles.archiveLabel}>내 회고</Text>
            </Pressable>
          ) : <View style={styles.backPlaceholder} />}
        </View>
        <ScrollView contentContainerStyle={styles.content}>
          {children}
          {action ? <View style={styles.actions}>{action}</View> : null}
        </ScrollView>
      </View>
    </View>
  );
}

function Home({ start, archive }) {
  return (
    <Shell title="올해 우리는" onArchive={archive}>
      <View style={styles.cover}>
        <Text style={styles.year}>2026</Text>
        <Text style={styles.coverLight} accessibilityElementsHidden importantForAccessibility="no">✦</Text>
        <Text accessibilityRole="header" style={styles.coverTitle}>{'사진으로 돌아보는\n올해의 장면들'}</Text>
        <Image source={art.winter} style={styles.window} resizeMode="contain" accessible={false} />
        <Text style={styles.winter}>2026년 겨울</Text>
      </View>
      <Pressable accessibilityRole="button" accessibilityLabel="잇기, 친구와 함께한 순간 찾아보기"
        onPress={start} style={({ pressed }) => [styles.entry, pressed && styles.pressed]}>
        <Pixel source={art.clayHeart} width={35} height={30} />
        <View style={styles.flex}>
          <Text style={styles.entryTitle}>잇기</Text>
          <Text style={styles.note}>친구와 함께한 순간 찾아보기</Text>
        </View>
        <Text style={styles.chevron}>{'›'}</Text>
      </Pressable>
    </Shell>
  );
}

function Friends({ invited = false }) {
  return (
    <View style={styles.friends}>
      <View style={styles.person}>
        <Pixel source={art.host} width={63} height={72} />
        <Text style={styles.friendName}>서연</Text>
      </View>
      <Pixel source={art.goldHeart} width={26} height={22} />
      <View style={styles.person}>
        <Pixel source={art.guest} width={63} height={72} />
        <Text style={styles.friendName}>{invited ? '지민' : '친구 자리'}</Text>
      </View>
    </View>
  );
}

function Photo({ photo, style }) {
  return <Image source={photos[photo.asset]} style={style} accessibilityLabel={photo.caption} resizeMode="cover" />;
}

function PhotoPair({ photos: pair, recap = false }) {
  return (
    <View style={styles.pair}>
      {pair.map(photo => {
        const name = photo.participantId === hostId ? '서연' : '지민';
        return (
          <View key={photo.photoId} style={styles.flex}>
            <Photo photo={photo} style={styles.pairPhoto} />
            <Text style={styles.photoOwner}>{name}{recap ? '의 시선' : '의 사진'}</Text>
          </View>
        );
      })}
    </View>
  );
}

function StatusRow({ name, status, ready }) {
  return (
    <View style={styles.statusRow}>
      <Text style={styles.entryTitle}>{name}</Text>
      <Text style={[styles.status, ready && styles.statusReady]}>{status}</Text>
    </View>
  );
}

export default function App() {
  const [state, dispatch] = useReducer(updateDemo, initialDemo);
  const { step, photoIds } = state;
  const { candidates, recap } = selectDemoView(state);
  const candidate = candidates.items[0];
  const transition = nextDemoStep[step];

  useEffect(() => {
    if (!transition) return;
    const [type, delay] = transition;
    const timeout = setTimeout(() => dispatch({ type }), delay);
    return () => clearTimeout(timeout);
  }, [transition]);

  const go = type => () => dispatch({ type });
  const end = go('reset');
  const pick = go('pick');
  const shell = { onBack: end };

  switch (step) {
    case 'invite':
      return (
        <Shell {...shell} key={step} action={<Button label="예시 방 시작하기" onPress={pick} />}>
          <Intro title={'누구와 올해를\n돌아볼까요?'} body="친구 한 명과 각자의 사진에서 함께 보낸 순간을 찾아요." />
          <Friends />
          <View style={styles.code}><Text style={styles.note}>예시 초대 코드</Text><Text style={styles.codeValue}>YE14</Text></View>
          <Text style={styles.note}>예시 방에서 친구와 회고를 만드는 과정을 체험해요.</Text>
        </Shell>
      );
    case 'pick':
      return (
        <Shell {...shell} key={step} action={<Button label="분석에 동의하고 제출" onPress={go('submit')} disabled={photoIds.length < 10} />}>
          <Intro title={'올해를 담은\n사진을 골라요'} body="예시 앨범에서 친구와 같은 날 찍은 사진을 10장 이상 골라 주세요." />
          <View style={styles.gridHeader}>
            <Text style={styles.entryTitle}>내가 고른 사진</Text>
            <Text style={styles.note} accessibilityLiveRegion="polite">{photoIds.length}장 선택</Text>
          </View>
          <View style={styles.photoGrid}>
            {catalog.map((photo, index) => {
              const checked = photoIds.includes(photo.photoId);
              return (
                <Pressable key={photo.photoId} accessibilityRole="checkbox" accessibilityState={{ checked }}
                  accessibilityLabel={index + 1 + '번째 사진, ' + photo.caption}
                  onPress={() => dispatch({ type: 'togglePhoto', photoId: photo.photoId })} style={styles.tile}>
                  <Photo photo={photo} style={styles.thumbnail} />
                  <Text style={[styles.check, checked && styles.checked]} accessible={false}>{checked ? '✓' : ''}</Text>
                </Pressable>
              );
            })}
          </View>
          <View style={styles.notice}><Text style={styles.noticeText}>이 사진으로 함께한 순간을 찾아요. 친구에게 보여 줄 사진은 나중에 골라요.</Text></View>
        </Shell>
      );
    case 'waiting':
    case 'analyzing':
      return (
        <Shell {...shell} key={step} action={<Button label="홈으로 돌아가기" onPress={end} secondary />}>
          <Intro title={step === 'waiting' ? '지민이 사진을\n고르고 있어요' : '함께 보낸 순간을\n찾고 있어요'}
            body={step === 'waiting' ? '두 사람의 사진이 모두 준비되면 분석을 시작해요.' : '두 사람이 찍은 예시 사진을 비교하고 있어요.'} />
          <Friends invited />
          <View accessibilityLiveRegion="polite">
            <StatusRow name="서연 (나)" status="제출 완료" ready />
            <StatusRow name="지민" status={step === 'waiting' ? '사진 선택 중' : '제출 완료'} ready={step === 'analyzing'} />
          </View>
        </Shell>
      );
    case 'approval':
    case 'visibilityWait':
      return (
        <Shell {...shell} key={step} action={step === 'approval' ? (
          <Button label="이 사진 보여 주기" onPress={go('approve')} />
        ) : <Button label="홈으로 돌아가기" onPress={end} secondary />}>
          <Intro title={step === 'approval' ? '지민에게 보여 줄\n사진을 확인해요' : '지민의 사진 공개를\n기다리고 있어요'}
            body={step === 'approval' ? '함께 보낸 순간으로 보이는 내 사진이에요.' : '각자 승인한 사진을 함께 볼 수 있어요.'} />
          {candidate.viewerOwnedPhotos.map(photo => (
            <View style={styles.approvalPhoto} key={photo.photoId}>
              <Photo photo={photo} style={styles.pairPhoto} />
              <Text style={styles.photoOwner}>12월 14일 · 서연의 사진</Text>
            </View>
          ))}
          <View style={styles.notice}><Text style={styles.noticeText}>승인한 사진은 이 방의 친구에게 보여요.</Text></View>
        </Shell>
      );
    case 'compare':
    case 'confirmationWait':
      return (
        <Shell {...shell} key={step} action={step === 'compare' ? (
          <View style={styles.pair}>
            <Button label="다른 순간이에요" onPress={go('reject')} secondary flex />
            <Button label="같은 순간이에요" onPress={go('confirm')} flex />
          </View>
        ) : <Button label="홈으로 돌아가기" onPress={end} secondary />}>
          <Intro title={step === 'compare' ? '같은 날의\n기억인가요?' : '지민의 답을\n기다리고 있어요'}
            body={step === 'compare' ? '둘 다 보여 주기로 한 사진이에요. 각자 같은 순간인지 확인해요.' : '지민도 같은 순간이라고 확인하면 회고에 담겨요.'} />
          <View style={styles.gridHeader}><Text style={styles.entryTitle}>12월 14일</Text></View>
          <PhotoPair photos={candidate.photos} />
          {step === 'confirmationWait' ? <View accessibilityLiveRegion="polite">
            <StatusRow name="서연 (나)" status="확인 완료" ready />
            <StatusRow name="지민" status="확인 중" />
          </View> : null}
        </Shell>
      );
    case 'empty':
      return (
        <Shell {...shell} key={step} action={<Button label="사진 다시 고르기" onPress={pick} />}>
          <Intro title={'함께한 순간을\n찾지 못했어요'} body="친구와 같은 날을 기억해 보고 사진을 다시 골라 보세요." />
          <Friends invited />
        </Shell>
      );
    case 'recap':
    case 'archive':
      if (!recap) return (
        <Shell {...shell} title="내 회고" action={<Button label="친구와 시작하기" onPress={go('invite')} />}>
          <Intro title="아직 담긴 회고가 없어요" body="친구와 같은 순간을 확인하면 여기에 회고가 생겨요." />
        </Shell>
      );
      return (
        <Shell {...shell} dark action={<Button label="홈으로 돌아가기" onPress={end} secondary />}>
          <Text style={styles.year}>2026</Text>
          <Intro dark title="함께 보낸 올해" body="서연과 지민이 함께 남긴 순간" />
          {recap.items.map(item => (
            <View key={item.candidateId} style={styles.recapCard}>
              <PhotoPair photos={item.photos} recap />
              <View style={styles.recapCaption}>
                <Text style={styles.note}>12월 14일</Text>
                <Text style={styles.caption}>{'같은 테이블,\n서로 다른 두 장'}</Text>
              </View>
            </View>
          ))}
        </Shell>
      );
    default:
      return <Home start={go('invite')} archive={go('openArchive')} />;
  }
}

const styles = StyleSheet.create({
  outer: { flex: 1, alignItems: 'center', backgroundColor: '#E8E1D5' },
  darkOuter: { backgroundColor: '#24354E' },
  screen: { flex: 1, width: '100%', maxWidth: 460, backgroundColor: color.paper },
  darkScreen: { backgroundColor: color.night },
  top: { height: 62, paddingHorizontal: 18, flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between' },
  homeTop: { paddingHorizontal: 24, gap: 10 },
  navTitle: { fontSize: 15, fontWeight: '700', color: color.ink },
  homeName: { flex: 1, fontSize: 17 },
  back: { width: 44, height: 44, alignItems: 'center', justifyContent: 'center' },
  backText: { fontSize: 29, color: color.ink, lineHeight: 36 },
  backPlaceholder: { width: 44 },
  archiveButton: { minHeight: 44, minWidth: 54, justifyContent: 'center', alignItems: 'flex-end' },
  archiveLabel: { color: color.muted, fontWeight: '600', fontSize: 14 },
  content: { flexGrow: 1, paddingHorizontal: 24, paddingTop: 14, paddingBottom: 28, gap: 20 },
  flex: { flex: 1, minWidth: 0 },
  intro: { paddingTop: 17, marginBottom: 10, gap: 12 },
  heading: { fontSize: 32, lineHeight: 42, fontWeight: '700', color: color.ink, letterSpacing: -0.9 },
  description: { fontSize: 15, lineHeight: 24, color: color.muted },
  paperText: { color: color.white },
  blueText: { color: color.blue },
  note: { fontSize: 13, lineHeight: 20, color: color.muted },
  cover: { backgroundColor: color.navy, padding: 19, borderRadius: 10, gap: 12 },
  year: { fontFamily: 'Arial', fontSize: 68, lineHeight: 76, letterSpacing: -3, fontWeight: '700', color: color.gold },
  coverLight: { position: 'absolute', right: 24, top: 28, color: color.blue, fontSize: 25 },
  coverTitle: { fontSize: 28, lineHeight: 37, color: color.white, fontWeight: '700', letterSpacing: -0.7 },
  window: { width: '100%', aspectRatio: 48 / 28, marginTop: 13 },
  winter: { color: color.blue, fontSize: 13, paddingBottom: 2 },
  entry: { flexDirection: 'row', alignItems: 'center', gap: 15, borderWidth: 1, borderColor: color.line, borderRadius: 8, padding: 15, backgroundColor: color.white },
  entryTitle: { fontSize: 16, fontWeight: '700', color: color.ink },
  chevron: { fontSize: 21, color: color.muted },
  friends: { minHeight: 196, backgroundColor: color.navy, borderRadius: 10, alignItems: 'center', justifyContent: 'space-evenly', flexDirection: 'row', gap: 8, paddingVertical: 24, paddingHorizontal: 9 },
  person: { alignItems: 'center', gap: 12 },
  friendName: { color: color.white, fontSize: 14, fontWeight: '700' },
  code: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', borderWidth: 1, borderColor: color.line, borderRadius: 8, padding: 15, backgroundColor: color.white },
  codeValue: { color: color.ink, fontSize: 19, fontWeight: '700', letterSpacing: 1 },
  gridHeader: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center' },
  photoGrid: { flexDirection: 'row', flexWrap: 'wrap', justifyContent: 'space-between', rowGap: 6 },
  tile: { width: '24%', aspectRatio: 1.12, borderRadius: 6, overflow: 'hidden' },
  thumbnail: { width: '100%', height: '100%' },
  check: { position: 'absolute', right: 5, top: 5, height: 22, width: 22, backgroundColor: color.white, textAlign: 'center', lineHeight: 20, color: color.white, borderColor: color.white, borderWidth: 1, borderRadius: 4 },
  checked: { backgroundColor: color.navy, fontWeight: '700' },
  notice: { backgroundColor: color.soft, borderRadius: 7, paddingHorizontal: 14, paddingVertical: 12 },
  noticeText: { color: color.ink, fontSize: 13, lineHeight: 19 },
  statusRow: { minHeight: 47, flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', borderBottomWidth: 1, borderBottomColor: color.line },
  status: { fontSize: 12, fontWeight: '600', color: color.navy, paddingHorizontal: 10, paddingVertical: 5, borderRadius: 5, backgroundColor: color.soft },
  statusReady: { color: color.sageInk, backgroundColor: color.sage },
  approvalPhoto: { width: '58%' },
  pair: { flexDirection: 'row', gap: 10 },
  pairPhoto: { width: '100%', aspectRatio: 0.76, borderRadius: 6, backgroundColor: color.soft },
  photoOwner: { color: color.ink, fontWeight: '600', fontSize: 13, paddingTop: 10 },
  recapCard: { borderRadius: 9, backgroundColor: color.white, padding: 12 },
  recapCaption: { marginTop: 17, paddingTop: 14, borderTopWidth: 1, borderTopColor: color.line, gap: 5 },
  caption: { color: color.ink, fontWeight: '700', fontSize: 21, lineHeight: 29 },
  actions: { marginTop: 'auto', paddingTop: 10 },
  button: { minHeight: 52, borderRadius: 8, backgroundColor: color.navy, justifyContent: 'center', alignItems: 'center', paddingHorizontal: 8, paddingVertical: 10 },
  secondary: { backgroundColor: color.white, borderWidth: 1, borderColor: color.line },
  buttonLabel: { fontSize: 14, fontWeight: '700', lineHeight: 20, color: color.white, textAlign: 'center' },
  secondaryLabel: { color: color.ink },
  pressed: { opacity: 0.75 },
  disabled: { backgroundColor: '#647284' },
});
