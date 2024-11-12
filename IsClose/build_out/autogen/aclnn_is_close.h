
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_IS_CLOSE_H_
#define ACLNN_IS_CLOSE_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnIsCloseGetWorkspaceSize
 * parameters :
 * x1 : required
 * x2 : required
 * rtolOptional : optional
 * atolOptional : optional
 * equalNanOptional : optional
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnIsCloseGetWorkspaceSize(
    const aclTensor *x1,
    const aclTensor *x2,
    double rtolOptional,
    double atolOptional,
    bool equalNanOptional,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnIsClose
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnIsClose(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    const aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
