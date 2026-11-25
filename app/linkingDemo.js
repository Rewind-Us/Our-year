export const hostId = 'p-a';
export const friendId = 'p-b';
export const catalog = [
  ['photo-a-03', 'cafe', '카페에서 카메라를 든 지민'],
  ['photo-a-04', 'tea', '테이블 위 케이크와 차'],
  ['photo-a-05', 'snow', '눈이 쌓인 거리'],
  ['photo-a-06', 'snow', '겨울 산책'],
  ['photo-a-07', 'cafe', '창가에서 찍은 사진'],
  ['photo-a-08', 'tea', '따뜻한 차 한 잔'],
  ['photo-a-09', 'tea', '겨울 카페 테이블'],
  ['photo-a-10', 'snow', '눈 내린 저녁'],
  ['photo-a-11', 'cafe', '카메라를 든 친구'],
  ['photo-a-12', 'tea', '친구와 나눈 디저트'],
  ['photo-a-13', 'cafe', '카페에 앉은 친구'],
  ['photo-a-14', 'tea', '마지막 한 모금'],
].map(([photoId, asset, caption]) => ({ photoId, asset, caption }));
const partnerPhoto = { photoId: 'photo-b-07', asset: 'friend', caption: '지민이 찍은 서연' };
const candidateId = 'event-demo';
export const initialDemo = {
  step: 'home',
  photoIds: catalog.filter((_, index) => index !== 2 && index !== 7).map(photo => photo.photoId),
  recap: null,
};

export function updateDemo(state, action) {
  switch (action.type) {
    case 'reset':
      return { ...initialDemo, recap: state.recap };
    case 'openArchive':
      return { ...state, step: 'archive' };
    case 'invite':
      return { ...initialDemo, step: 'invite', recap: state.recap };
    case 'pick':
      return { ...state, step: 'pick' };
    case 'togglePhoto': {
      if (state.step !== 'pick') return state;
      const photoIds = state.photoIds.includes(action.photoId)
        ? state.photoIds.filter(id => id !== action.photoId)
        : [...state.photoIds, action.photoId];
      return { ...state, photoIds };
    }
    case 'submit':
      return state.step === 'pick' && state.photoIds.length >= 10 ? { ...state, step: 'waiting' } : state;
    case 'friendSubmitted':
      return state.step === 'waiting' ? { ...state, step: 'analyzing' } : state;
    case 'analysisReady':
      if (state.step !== 'analyzing') return state;
      return { ...state, step: state.photoIds.includes('photo-a-03') ? 'approval' : 'empty' };
    case 'approve':
      return state.step === 'approval' ? { ...state, step: 'visibilityWait' } : state;
    case 'friendApproved':
      return state.step === 'visibilityWait' ? { ...state, step: 'compare' } : state;
    case 'reject':
      return state.step === 'compare' ? { ...state, step: 'empty' } : state;
    case 'confirm':
      return state.step === 'compare' ? { ...state, step: 'confirmationWait' } : state;
    case 'friendConfirmed': {
      if (state.step !== 'confirmationWait') return state;
      const photos = selectDemoView(state).candidates.items[0].photos;
      return { ...state, step: 'recap', recap: { roomId: 'room-demo', items: [
        { candidateId, eventDate: '2026-12-14', photos },
      ] } };
    }
    default:
      return state;
  }
}

export function selectDemoView(state) {
  const { step, recap } = state;
  const roomStatus = {
    analyzing: 'analyzing',
    approval: 'reviewing',
    visibilityWait: 'reviewing',
    compare: 'reviewing',
    confirmationWait: 'reviewing',
    empty: 'no_candidates',
    recap: 'recap_ready',
  }[step] ?? 'awaiting_submission';
  const participants = [
    { id: hostId, submissionStatus: ['home', 'invite', 'pick', 'archive'].includes(step) ? 'pending' : 'submitted' },
    { id: friendId, submissionStatus: ['home', 'invite', 'pick', 'archive', 'waiting'].includes(step) ? 'pending' : 'submitted' },
  ];
  const room = { roomId: 'room-demo', viewerId: hostId, roomStatus, participants,
    jobId: participants.every(person => person.submissionStatus === 'submitted') ? 'job-demo' : null };
  if (step === 'approval' || step === 'visibilityWait') {
    return { room, candidates: { items: [{
      candidateId, candidateStatus: 'awaiting_visibility', viewerOwnedPhotos: [catalog[0]],
      viewerVisibility: step === 'visibilityWait' ? 'approved' : 'pending', partnerVisibility: 'pending',
    }] } };
  }
  if (step === 'compare' || step === 'confirmationWait') {
    return { room, candidates: { items: [{
      candidateId, candidateStatus: 'awaiting_confirmation',
      viewerConfirmation: step === 'confirmationWait' ? 'same_moment' : 'pending', partnerConfirmation: 'pending',
      photos: [{ participantId: hostId, ...catalog[0] }, { participantId: friendId, ...partnerPhoto }],
    }] } };
  }
  return { room, candidates: { items: [] }, recap: ['archive', 'recap'].includes(step) ? recap : null };
}
