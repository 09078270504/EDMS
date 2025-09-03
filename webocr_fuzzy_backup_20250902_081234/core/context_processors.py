from .models import ChatConversation

def chat_history(request):
    """Add chat conversations to all templates"""
    conversations = []
    current_conversation = None
    
    if request.user.is_authenticated:
        conversations = ChatConversation.objects.filter(user=request.user).order_by('-created_at')  # All conversations, latest first
        
        # Check if we're viewing a specific conversation
        if request.resolver_match and 'conversation_id' in request.resolver_match.kwargs:
            conversation_id = request.resolver_match.kwargs['conversation_id']
            try:
                current_conversation = ChatConversation.objects.get(id=conversation_id, user=request.user)
            except ChatConversation.DoesNotExist:
                pass
    
    return {
        'conversations': conversations,
        'current_conversation': current_conversation,
    }
