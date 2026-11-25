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
const demoCandidates = [
  { candidateId: 'event-demo', eventDate: '2026-12-14', viewerPhotoId: 'photo-a-03', caption: '같은 테이블,\n서로 다른 두 장',
    partnerPhoto: { photoId: 'photo-b-07', asset: 'friend', caption: '지민이 찍은 서연' } },
  { candidateId: 'event-demo-2', eventDate: '2026-12-20', viewerPhotoId: 'photo-a-12', caption: '함께 보낸\n겨울 하루',
    partnerPhoto: { photoId: 'photo-b-11', asset: 'snow', caption: '지민이 찍은 눈 내린 거리' } },
];
const findCandidate = id => demoCandidates.find(item => item.candidateId === id);
const candidatePhotos = item => [
  { participantId: hostId, ...catalog.find(photo => photo.photoId === item.viewerPhotoId) },
  { participantId: friendId, ...item.partnerPhoto },
];
export const initialDemo = {
  step: 'home',
  photoIds: catalog.filter((_, index) => index !== 2 && index !== 7).map(photo => photo.photoId),
  queue: [],
  confirmed: [],
  recap: null,
};

// 지금 후보의 검토를 마치고 다음 후보로 가요. 남은 후보가 없으면 확인한 순간으로 회고를 만들어요.
function nextCandidate(state, confirmed = state.confirmed) {
  const queue = state.queue.slice(1);
  if (queue.length) return { ...state, queue, confirmed, step: 'approval' };
  if (!confirmed.length) return { ...state, queue, confirmed, step: 'allDeclined' };
  const items = confirmed.map(id => {
    const item = findCandidate(id);
    return { candidateId: id, eventDate: item.eventDate, caption: item.caption, photos: candidatePhotos(item) };
  });
  return { ...state, queue, confirmed, step: 'recap', recap: { roomId: 'room-demo', items } };
}

export function updateDemo(state, action) {
  switch (action.type) {
    case 'reset':
      return { ...initialDemo, recap: state.recap };
    case 'openArchive':
      return { ...state, step: 'archive' };
    case 'invite':
      return { ...initialDemo, step: 'invite', recap: state.recap };
    case 'pick':
      return { ...state, step: 'pick', queue: [], confirmed: [] };
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
    case 'analysisReady': {
      if (state.step !== 'analyzing') return state;
      const queue = demoCandidates.filter(item => state.photoIds.includes(item.viewerPhotoId)).map(item => item.candidateId);
      return { ...state, queue, confirmed: [], step: queue.length ? 'approval' : 'empty' };
    }
    case 'analysisFailed':
      return state.step === 'analyzing' ? { ...state, step: 'failed' } : state;
    case 'decline':
      return state.step === 'approval' ? nextCandidate(state) : state;
    case 'approve':
      return state.step === 'approval' ? { ...state, step: 'visibilityWait' } : state;
    case 'friendApproved':
      return state.step === 'visibilityWait' ? { ...state, step: 'compare' } : state;
    case 'reject':
      return state.step === 'compare' ? nextCandidate(state) : state;
    case 'confirm':
      return state.step === 'compare' ? { ...state, step: 'confirmationWait' } : state;
    case 'friendConfirmed':
      return state.step === 'confirmationWait' ? nextCandidate(state, [...state.confirmed, state.queue[0]]) : state;
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
    allDeclined: 'reviewing',
    empty: 'no_candidates',
    failed: 'analysis_failed',
    recap: 'recap_ready',
  }[step] ?? 'awaiting_submission';
  const participants = [
    { id: hostId, submissionStatus: ['home', 'invite', 'pick', 'archive'].includes(step) ? 'pending' : 'submitted' },
    { id: friendId, submissionStatus: ['home', 'invite', 'pick', 'archive', 'waiting'].includes(step) ? 'pending' : 'submitted' },
  ];
  const room = { roomId: 'room-demo', viewerId: hostId, roomStatus, participants,
    jobId: participants.every(person => person.submissionStatus === 'submitted') ? 'job-demo' : null };
  const current = findCandidate(state.queue[0]);
  if (step === 'approval' || step === 'visibilityWait') {
    return { room, candidates: { items: [{
      candidateId: current.candidateId, eventDate: current.eventDate, candidateStatus: 'awaiting_visibility',
      viewerOwnedPhotos: candidatePhotos(current).filter(photo => photo.participantId === hostId),
      viewerVisibility: step === 'visibilityWait' ? 'approved' : 'pending', partnerVisibility: 'pending',
    }] } };
  }
  if (step === 'compare' || step === 'confirmationWait') {
    return { room, candidates: { items: [{
      candidateId: current.candidateId, eventDate: current.eventDate, candidateStatus: 'awaiting_confirmation',
      viewerConfirmation: step === 'confirmationWait' ? 'same_moment' : 'pending', partnerConfirmation: 'pending',
      photos: candidatePhotos(current),
    }] } };
  }
  if (step === 'failed') {
    return { room, job: { jobId: 'job-demo', status: 'failed', error: { code: 'ANALYSIS_FAILED', retryable: true } }, candidates: { items: [] } };
  }
  return { room, candidates: { items: [] }, recap: ['archive', 'recap'].includes(step) ? recap : null };
}
