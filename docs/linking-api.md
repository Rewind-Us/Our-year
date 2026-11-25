# 잇기 API 계약 초안

친구 두 명이 각자 고른 사진에서 같이 보낸 순간을 찾아 회고에 담아요. 프런트는 [가상 응답](../data/schema/linking-api-examples.json)을 불러와 화면을 만들고, 서버는 같은 상태와 응답 구조를 기준으로 구현을 시작할 수 있어요. 경로와 필드 이름은 연동을 위한 제안이에요.

## 화면과 상태

새로 접속하거나 페이지를 새로고침하면 방을 조회해 이어서 볼 화면을 정해요. `viewerId`는 서버가 인증된 참여자를 기준으로 반환해요. 홈과 내 회고는 앱 공통 화면이고, 아래 계약은 잇기에서 사용해요.

| 화면 | 응답에서 확인할 값 | 가능한 행동 |
| --- | --- | --- |
| 친구 초대, 사진 선택 | `roomStatus: awaiting_submission`, 내 `submissionStatus: pending` | 초대 링크 보내기, 사진 선택하고 제출하기 |
| 친구 제출 대기 | `awaiting_submission`, 내 상태는 `submitted`, 친구는 `pending` | 같은 링크로 돌아와 상태 조회 |
| 분석 중 | `analyzing`, 작업 `queued` 또는 `processing` | 방과 작업 상태 조회 |
| 내 사진 공개 | `reviewing`, 후보 `awaiting_visibility` | 내가 보여 줄 사진 승인, 친구 승인 대기 |
| 같은 순간 확인 | `reviewing`, 후보 `awaiting_confirmation` | `same_moment` 또는 `different_moment` 선택 |
| 친구 확인 대기 | 내 응답은 `same_moment`, 친구 응답은 `pending` | 후보 상태 조회 |
| 공동 회고 | `recap_ready` | 두 사람이 같은 순간이라고 확인한 후보 보기 |
| 검색 결과 없음, 분석 실패 | `no_candidates` 또는 `analysis_failed` | 사진 다시 고르기, 실패 안내와 재시도 |

## 요청과 응답

| 요청 | 보내는 값 | 받아서 쓰는 값 |
| --- | --- | --- |
| `POST /rooms` | 빈 JSON 객체 | `roomId`, `inviteCode`, 생성자 참여 세션 |
| `POST /rooms/{id}/join` | `inviteCode` | 초대받은 사람의 참여 세션과 방 상태 |
| `GET /rooms/{id}` | 참여 세션 | `roomStatus`, `viewerId`, `participants`, `jobId` |
| `POST /rooms/{id}/photos` | 분석 동의 후 사진별 `clientPhotoId`, 축소본, `capturedAt` | 내 `photoId`. 실패한 파일만 다시 올릴 때 같은 `clientPhotoId` 사용 |
| `POST /rooms/{id}/submit` | `photoIds` | 최신 방 응답. 두 번째 제출이 끝나면 분석 작업의 `jobId` 포함 |
| `GET /jobs/{jobId}` | 참여 세션 | `queued`, `processing`, `succeeded`, `failed`와 실패 시 `error` |
| `GET /rooms/{id}/candidates` | 참여 세션 | `items`에 후보 ID, 후보별 상태, 현재 공개 범위의 사진 |
| `PATCH /photos/{id}/visibility` | `candidateId`, `approved` | 해당 후보의 공개 상태 |
| `POST /events/{id}/confirmation` | `decision` | 해당 후보의 확인 상태 |
| `GET /rooms/{id}/recap` | 참여 세션 | 두 사람이 확인을 마친 후보로 만든 회고 `items` |

촬영 시각은 시간대가 포함된 ISO 8601 형식으로 전달해요. 시각을 확인할 수 없으면 `null`을 보냅니다. 초대 코드는 참여에 쓰고 이후 요청은 별도 참여 세션으로 인증해요. 세션 전달 방식, 분석 실패 후 재시도 경로, 사진 보관 및 삭제 기준은 실제 연결 전에 합의해요.

## 화면 전환을 결정하는 세 가지 조건

1. **두 번째 제출이 끝나면 분석을 시작해요.** 서버가 분석 작업을 한 번만 등록해요. 같은 사진 목록을 재전송하면 기존 제출과 작업 ID를 반환해요. 두 제출이 거의 동시에 도착하는 경우에도 방 상태와 작업 등록을 함께 처리해요.
2. **분석 동의 후 축소본을 보내고, 공개 승인으로 친구에게 보여 줘요.** 분석 단계에서는 원본 없이 축소본과 촬영 시각을 다뤄요. 후보 공개 전에는 `viewerOwnedPhotos`에 내 사진만 들어가요. 서버는 상대 사진 ID와 미리보기 경로를 응답에서 제외해요. 후보의 양쪽 사진이 각 소유자의 승인을 받으면 `photos`에 두 사람의 사진을 담아요. 미리보기 접근 권한도 같은 기준으로 확인해요.
3. **두 사람이 같은 순간이라고 답하면 회고에 들어가요.** 확인은 참여자와 후보별로 기록하고 같은 응답을 재전송해도 중복으로 기록하지 않아요. 한 사람이 `different_moment`를 선택하면 해당 후보는 회고에서 제외해요. 다른 후보가 있으면 다음 후보로 이동할 수 있어요.

## 실패와 빈 상태

- 아직 분석 결과가 없으면 기다리는 화면을 유지해요. 분석이 완료됐고 후보가 없을 때는 `no_candidates`와 빈 `items`로 사진 다시 고르기를 보여 줘요.
- 사진 업로드가 일부 실패하면 선택을 그대로 두고 실패한 파일만 다시 올려요. 방의 참여자나 사진 소유자가 아니면 변경을 거절해요.
- 분석 실패는 `analysis_failed`와 `error.code`로 구분해요. 알 수 없는 상태는 안전하게 다시 조회할 수 있는 안내를 보여 줘요.

## 가상 응답으로 확인하기

예시 파일의 각 항목은 화면 한 상태를 재현해요. `room`, `job`, `candidates`, `recap`은 해당 API 응답을 나타내고, 사진 ID와 미리보기 경로는 가상 값이에요. `awaitingVisibility`에서 친구 사진이 노출되지 않는지 확인한 뒤 `awaitingConfirmation`과 `ready`에서 두 사람의 확인 상태와 회고 카드가 달라지는지 확인해요.
